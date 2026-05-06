"""
Depth-based fallback obstacle proposer.

Given a metric depth map and the YOLO detection masks already covering
known objects, proposes additional "unclassified obstacle" regions that
are physically close to the camera but were NOT detected by YOLO. This
fills the gap for COCO-not-in-classes things (walls, partial doorframes,
random clutter) that the visually-impaired user still needs to know about.

Public API:
  - propose_obstacles(depth_map, claimed_masks=None, ...) → List[Dict]

Each proposal is shaped like an `analyze_scene()` detection so it can be
appended to the result list and flow through the existing
proximity_alerter / hazard_scorer downstream.

Algorithm:
  1. Mark pixels claimed by existing YOLO masks → set their depth to +inf.
  2. Build close_mask = (depth < threshold_m) on the unclaimed pixels.
  3. Connected-components label the close_mask.
  4. For each component:
       * area filter (drop tiny / spammy ones)
       * compute centroid, bbox, top_k closest depth → distance_m
       * compute direction + angle via the same FOV math as scene_analyzer
  5. Return as detection-shaped dicts with class_name="obstacle".

This module is a single file with no Ultralytics dependency. It only needs
NumPy (and uses cv2.connectedComponentsWithStats — already a project dep).
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.scene_analyzer import _compute_direction_and_angle

logger = logging.getLogger(__name__)


# ============================================================
# Defaults (mirror docs/PLAN_DEPTH_AWARE_YOLO.md §6.3)
# ============================================================

DEFAULT_DISTANCE_THRESHOLD_M = 2.0      # only pixels closer than this are candidates
DEFAULT_MIN_AREA_FRAC = 0.005           # 0.5% of image — drop noise
DEFAULT_MAX_PROPOSALS = 5               # top-K by closeness
DEFAULT_TOP_K_PIXELS = 100              # for representative distance, mean of K closest


# ============================================================
# Helpers
# ============================================================


def _claimed_mask_union(
    masks: Iterable[Optional[np.ndarray]],
    shape: Tuple[int, int],
) -> np.ndarray:
    """Boolean OR of all per-detection masks; returns False everywhere if list is empty."""
    H, W = shape
    union = np.zeros((H, W), dtype=bool)
    for m in masks or ():
        if m is None:
            continue
        m_b = np.asarray(m, dtype=bool)
        if m_b.shape != (H, W):
            logger.warning(
                f"  obstacle_proposer: skipping mask with mismatched shape "
                f"{m_b.shape} vs depth {(H, W)}"
            )
            continue
        union |= m_b
    return union


def _component_distance_m(depth_values: np.ndarray, top_k: int) -> float:
    """Mean of the K smallest depth values in a component (closest-K mean)."""
    if depth_values.size == 0:
        return float("nan")
    k = min(top_k, depth_values.size)
    closest = np.partition(depth_values, k - 1)[:k]
    return float(np.mean(closest))


# ============================================================
# Public API
# ============================================================


def propose_obstacles(
    depth_map: np.ndarray,
    claimed_masks: Optional[Iterable[Optional[np.ndarray]]] = None,
    *,
    distance_threshold_m: float = DEFAULT_DISTANCE_THRESHOLD_M,
    min_area_frac: float = DEFAULT_MIN_AREA_FRAC,
    max_proposals: int = DEFAULT_MAX_PROPOSALS,
    top_k_pixels: int = DEFAULT_TOP_K_PIXELS,
) -> List[Dict]:
    """
    Propose unclassified obstacle regions from a metric depth map.

    Args:
        depth_map: (H, W) float metric depth in meters. NaN/inf treated as
            "no signal" (excluded from proposals).
        claimed_masks: Iterable of per-object boolean masks (H, W) already
            covered by detections. These regions are excluded so we don't
            re-emit known objects as obstacles. If None, treats whole image
            as unclaimed.
        distance_threshold_m: Pixels with depth above this are not candidates.
        min_area_frac: Drop components smaller than this fraction of HxW.
        max_proposals: Return at most this many proposals (sorted by closeness).
        top_k_pixels: For each component, distance = mean of K closest pixels.

    Returns:
        List of detection-shaped dicts, sorted by distance (closest first):
            {
                "class_id":         -1,             # sentinel — not a YOLO class
                "class_name":       "obstacle",
                "confidence":        float,         # heuristic: 1 - normalized distance
                "bbox":              ndarray[4],    # [x1, y1, x2, y2] pixel coords
                "mask":              ndarray[H, W] bool,
                "mask_score":        1.0,           # not from a model — flag value
                "depth_stats":       {"top_k_100": ..., "mean": ..., "min": ..., "max": ...},
                "direction":         "left"|"center"|"right",
                "angle_deg":         float,
                "centroid_x_norm":   float,
                "source":            "obstacle_proposer",
            }
    """
    import cv2  # lazy: only here to keep module-level imports light

    if depth_map.ndim != 2:
        raise ValueError(f"depth_map must be 2-D, got shape {depth_map.shape}")

    H, W = depth_map.shape
    if H == 0 or W == 0:
        return []

    # Mask out unsignaled pixels and pixels claimed by YOLO detections.
    claimed = _claimed_mask_union(claimed_masks, (H, W))
    valid = np.isfinite(depth_map) & ~claimed

    # Candidate close pixels
    close_mask = valid & (depth_map < distance_threshold_m) & (depth_map > 0)
    if not close_mask.any():
        return []

    # 8-connectivity components
    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        close_mask.astype(np.uint8), connectivity=8
    )
    # label 0 is background

    min_area_px = max(1, int(min_area_frac * H * W))

    proposals: List[Dict] = []
    for lbl in range(1, n_labels):
        x, y, w, h, area = stats[lbl]
        if area < min_area_px:
            continue

        comp_mask = (labels == lbl)
        depth_values = depth_map[comp_mask]
        depth_values = depth_values[np.isfinite(depth_values)]
        if depth_values.size == 0:
            continue

        distance_m = _component_distance_m(depth_values, top_k_pixels)
        if not np.isfinite(distance_m):
            continue

        # Centroid in pixel coords (use mask, not stats centroid which is float
        # but identical here — we want consistency with scene_analyzer style).
        ys, xs = np.where(comp_mask)
        cx_pix = float(xs.mean())
        cy_pix = float(ys.mean())  # currently unused for direction; kept for clarity
        centroid_x_norm = float(np.clip(cx_pix / max(W, 1), 0.0, 1.0))

        direction, angle_deg = _compute_direction_and_angle(centroid_x_norm)

        # Confidence heuristic: closer → higher confidence in [0, 1].
        # Linear in (1 - d/threshold) clamped to [0, 1].
        conf = float(np.clip(1.0 - distance_m / max(distance_threshold_m, 1e-6), 0.0, 1.0))

        depth_stats = {
            "mean": float(np.mean(depth_values)),
            "min": float(np.min(depth_values)),
            "max": float(np.max(depth_values)),
            "top_k_100": distance_m,
        }

        proposals.append(
            {
                "class_id": -1,
                "class_name": "obstacle",
                "confidence": conf,
                "bbox": np.array([x, y, x + w, y + h], dtype=np.float32),
                "mask": comp_mask,
                "mask_score": 1.0,
                "depth_stats": depth_stats,
                "direction": direction,
                "angle_deg": angle_deg,
                "centroid_x_norm": centroid_x_norm,
                "source": "obstacle_proposer",
            }
        )

    # Sort by closeness, keep top-K
    proposals.sort(key=lambda d: d["depth_stats"]["top_k_100"])
    if max_proposals > 0:
        proposals = proposals[:max_proposals]
    return proposals


def merge_with_detections(
    yolo_detections: List[Dict],
    proposals: List[Dict],
) -> List[Dict]:
    """
    Convenience: append depth-based proposals after YOLO detections.

    The downstream proximity_alerter / hazard_scorer don't care about input
    order — they re-sort. So this is just a thin wrapper for code clarity.
    """
    return list(yolo_detections) + list(proposals)
