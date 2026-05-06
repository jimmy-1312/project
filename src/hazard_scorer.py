"""
Hazard Scorer: Rank scene_analyzer detections by risk for visually impaired users.

Given the per-object output of `analyze_scene()`, this module:
  1. Maps detected class names to our hazard taxonomy
     (stairs / glass_door / pillar / hanging_obstacle / low_obstacle / person).
  2. Computes a per-detection risk score combining
     class priority × proximity (depth) × normalized mask area.
  3. Returns the top-K most dangerous objects sorted by risk.

The output is intended to feed an audio/text alert layer
(e.g. "person 1.5m to your left, chair 0.8m ahead").

Public API:
  - map_to_hazard_class(class_name) -> Optional[str]
  - compute_risk_score(detection, image_shape) -> float
  - rank_hazards(detections, image_shape, top_k=None) -> List[Dict]
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)


# ============================================================
# Class mapping
# ============================================================


def map_to_hazard_class(class_name: str) -> Optional[str]:
    """
    Map a detected class name to our hazard taxonomy.

    Resolution order:
      1. If `class_name` is already a hazard class (fine-tuned YOLO output),
         return it as-is.
      2. Otherwise look up `config.COCO_TO_HAZARD` (vanilla YOLO fallback).
      3. Else return None (drop from hazard ranking).

    Args:
        class_name: Class name from detector. May be a COCO class
            (e.g. "person", "chair") or a hazard class
            (e.g. "stairs", "glass_door") if using fine-tuned YOLO.

    Returns:
        Hazard class name, or None if the class is not recognized as a hazard.
    """
    if class_name in config.HAZARD_PRIORITY:
        return class_name
    return config.COCO_TO_HAZARD.get(class_name)


# ============================================================
# Risk scoring
# ============================================================


def _proximity_score(distance_m: float) -> float:
    """
    Map distance in meters to a proximity score in (0, 1].

    Uses a soft 1 / (1 + d / d_half) curve so that:
      - distance = 0       → score = 1.0
      - distance = d_half  → score = 0.5
      - distance = ∞       → score → 0

    Args:
        distance_m: Distance in meters. Non-positive values are clamped.

    Returns:
        Proximity score in (0, 1].
    """
    d_half = config.HAZARD_PROXIMITY_HALF_M
    d = max(float(distance_m), 0.0)
    return 1.0 / (1.0 + d / d_half)


def _extract_distance_m(detection: Dict) -> Optional[float]:
    """
    Extract a single distance estimate (meters) from a detection's depth_stats.

    Preference order: top_k_100 (closest-K mean, robust) > min > mean.
    Returns None if no finite depth is available.
    """
    depth_stats = detection.get("depth_stats") or {}
    for key in ("top_k_100", "min", "mean"):
        val = depth_stats.get(key)
        if val is None:
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f) and f > 0:
            return f
    return None


def _mask_area_norm(detection: Dict, image_shape: Tuple[int, int]) -> float:
    """
    Normalized mask area in [0, 1]: pixels in mask / total image pixels.

    Falls back to bbox area when the mask is missing/empty.
    """
    H, W = image_shape[:2]
    total = float(H * W)
    if total <= 0:
        return 0.0

    mask = detection.get("mask")
    if mask is not None and hasattr(mask, "sum"):
        area = float(np.asarray(mask, dtype=bool).sum())
        if area > 0:
            return min(area / total, 1.0)

    bbox = detection.get("bbox")
    if bbox is not None:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        bbox_area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
        return min(bbox_area / total, 1.0)

    return 0.0


def compute_risk_score(
    detection: Dict,
    image_shape: Tuple[int, int],
) -> float:
    """
    Compute a scalar risk score for one detection.

        risk = priority^w_p * proximity^w_d * area_norm^w_a

    All three components are in (0, 1]; the weighted product stays in [0, 1].

    Args:
        detection: One enriched detection dict from `analyze_scene()`.
            Must contain `class_name`. Optionally `depth_stats`, `mask`, `bbox`.
        image_shape: (H, W) of the source image.

    Returns:
        Risk score in [0, 1]. Returns 0.0 if the class is not a hazard or
        if depth is unavailable (we can't reason about distance).
    """
    hazard_class = map_to_hazard_class(detection.get("class_name", ""))
    if hazard_class is None:
        return 0.0

    priority = config.HAZARD_PRIORITY.get(hazard_class, 0.0)
    if priority <= 0:
        return 0.0

    distance = _extract_distance_m(detection)
    if distance is None:
        # No depth → conservative: drop. Alerts without distance are useless.
        return 0.0
    proximity = _proximity_score(distance)

    area_norm = _mask_area_norm(detection, image_shape)
    if area_norm <= 0:
        return 0.0

    w_p = config.HAZARD_WEIGHT_PRIORITY
    w_d = config.HAZARD_WEIGHT_PROXIMITY
    w_a = config.HAZARD_WEIGHT_AREA

    score = (priority ** w_p) * (proximity ** w_d) * (area_norm ** w_a)
    return float(np.clip(score, 0.0, 1.0))


# ============================================================
# Top-K ranking
# ============================================================


def rank_hazards(
    detections: List[Dict],
    image_shape: Tuple[int, int],
    top_k: Optional[int] = None,
) -> List[Dict]:
    """
    Sort detections by descending risk score and keep the top-K.

    Each returned dict is the original detection dict with two extra keys:
      - "hazard_class": str — mapped hazard class name
      - "risk_score":   float — the computed risk in [0, 1]
      - "distance_m":   Optional[float] — distance used for scoring

    Args:
        detections: List of detection dicts from `analyze_scene()`.
        image_shape: (H, W) of the source image.
        top_k: Max number of hazards to return. Defaults to config.HAZARD_TOP_K.
            Pass 0 or negative to return all hazards (sorted).

    Returns:
        List of detection dicts (copies), sorted by risk desc, length ≤ top_k.
        Detections with risk_score == 0 are dropped.
    """
    if top_k is None:
        top_k = config.HAZARD_TOP_K

    enriched = []
    for det in detections:
        hazard_class = map_to_hazard_class(det.get("class_name", ""))
        if hazard_class is None:
            continue

        score = compute_risk_score(det, image_shape)
        if score <= 0:
            continue

        out = dict(det)  # shallow copy; mask/bbox arrays are shared
        out["hazard_class"] = hazard_class
        out["risk_score"] = score
        out["distance_m"] = _extract_distance_m(det)
        enriched.append(out)

    enriched.sort(key=lambda d: d["risk_score"], reverse=True)

    if top_k and top_k > 0:
        enriched = enriched[:top_k]

    return enriched


# ============================================================
# Convenience: short text alert (for logging / future TTS)
# ============================================================


def distance_to_steps(distance_m: float, step_length_m: Optional[float] = None) -> int:
    """
    Convert distance in meters to an approximate step count.

    Uses `config.STEP_LENGTH_M` (default 0.7 m) as the average adult stride.
    Always returns at least 1 (anything closer than half a step rounds up
    to "1 step" — visually impaired users care more about "very close" than
    sub-step precision).

    Args:
        distance_m: Distance in meters. Non-positive values return 1.
        step_length_m: Override step length (m). Defaults to config.STEP_LENGTH_M.

    Returns:
        Integer step count, ≥ 1.
    """
    if step_length_m is None:
        step_length_m = config.STEP_LENGTH_M
    if not np.isfinite(distance_m) or distance_m <= 0:
        return 1
    return max(1, int(round(distance_m / step_length_m)))


def format_alert(hazard: Dict, use_steps: bool = True) -> str:
    """
    Format a single ranked hazard as a short alert string.

    Examples:
        use_steps=True  → "stairs ~2 steps ahead, slightly left (-15°)"
        use_steps=False → "stairs 1.2m, slightly left (-15°)"

    Args:
        hazard: One dict from `rank_hazards()`.
        use_steps: If True, express distance in steps (rounded). If False,
            express in meters with one decimal.

    Returns:
        Human-readable alert string.
    """
    cls = hazard.get("hazard_class", hazard.get("class_name", "object"))
    dist = hazard.get("distance_m")
    direction = hazard.get("direction", "")
    angle = hazard.get("angle_deg")

    parts = [cls]
    if dist is not None and np.isfinite(dist):
        if use_steps:
            steps = distance_to_steps(dist)
            parts.append(f"~{steps} step{'s' if steps != 1 else ''} ahead")
        else:
            parts.append(f"{dist:.1f}m ahead")
    if direction:
        if angle is not None and np.isfinite(angle):
            parts.append(f"{direction} ({angle:+.0f}deg)")
        else:
            parts.append(direction)
    return ", ".join(parts)
