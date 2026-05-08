"""End-to-end scene analyzer.

Given an image + (detector, segmentor, depth_estimator), runs all three and
returns a list of per-object dicts with bbox / mask / depth_stats / direction
/ angle. Used by scripts/run.py and scripts/evaluate.py.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import config
from src.depth_estimator import _aggregate_values

_REQUIRED_DETECTION_KEYS = frozenset({
    "bbox", "mask", "class_id", "class_name", "confidence", "mask_score",
})


def _parse_aggregation_modes(modes) -> Dict[str, Dict]:
    """Normalize user spec → {key: {mode, k, p}}."""
    if not isinstance(modes, (list, tuple)):
        raise TypeError(f"aggregation_modes must be list/tuple, got {type(modes).__name__}")

    simple = {"mean", "median", "max", "min"}
    out: Dict[str, Dict] = {}
    for item in modes:
        if isinstance(item, str):
            if item not in simple:
                raise ValueError(f"unknown simple mode {item!r}")
            out[item] = {"mode": item, "k": None, "p": None}
        elif isinstance(item, dict):
            mode = item.get("mode")
            if mode == "top_k":
                k = item.get("k")
                if not (isinstance(k, int) and k > 0):
                    raise ValueError("top_k requires positive int k")
                out[f"top_k_{k}"] = {"mode": "top_k", "k": k, "p": None}
            elif mode == "top_p":
                p = item.get("p")
                if not (isinstance(p, (int, float)) and 0.0 < p <= 1.0):
                    raise ValueError("top_p requires p in (0, 1]")
                out[f"top_p_{p}"] = {"mode": "top_p", "k": None, "p": float(p)}
            else:
                raise ValueError(f"unknown mode {mode!r}")
        else:
            raise ValueError(f"items must be str or dict, got {type(item).__name__}")
    return out


def _extract_depth_in_region(depth_map: np.ndarray, mask: np.ndarray,
                              bbox: np.ndarray) -> np.ndarray:
    """Take depth values inside mask if non-empty, else inside bbox."""
    if mask is not None and np.any(mask):
        return depth_map[mask]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    H, W = depth_map.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return np.array([], dtype=np.float32)
    return depth_map[y1:y2, x1:x2].ravel()


def _compute_centroid_from_mask_or_bbox(mask: np.ndarray, bbox: np.ndarray,
                                         image_w: int) -> float:
    """Return centroid x normalized to [0,1]. Mask if available, else bbox center."""
    if mask is not None and np.any(mask):
        ys, xs = np.where(mask)
        cx = float(np.mean(xs))
    else:
        x1, _, x2, _ = bbox
        cx = (float(x1) + float(x2)) / 2.0
    return float(np.clip(cx / max(image_w, 1), 0.0, 1.0))


def _compute_direction_and_angle(centroid_x_norm: float) -> Tuple[str, float]:
    """centroid_x_norm in [0,1] → (direction, angle_deg)."""
    fov = float(getattr(config, "HORIZONTAL_FOV", 60.0))
    angle_deg = (centroid_x_norm - 0.5) * fov
    left = float(getattr(config, "DIR_LEFT", 0.33))
    right = float(getattr(config, "DIR_RIGHT", 0.67))
    if centroid_x_norm < left:
        direction = "left"
    elif centroid_x_norm > right:
        direction = "right"
    else:
        direction = "center"
    return direction, angle_deg


def analyze_scene(
    image: np.ndarray,
    detector,
    segmentor,
    depth_estimator,
    aggregation_modes: Optional[Union[List[str], List[Dict]]] = None,
    *,
    closest_side: str = "high",
    scale_depth_to_meters: bool = False,
) -> List[Dict]:
    """Run the 3-model pipeline on one image and return per-object dicts.

    closest_side: 'high' (relative depth: bigger=closer) or 'low' (metric: smaller=closer).
    scale_depth_to_meters: if True, scale relative depth → meters and auto-flip closest_side.
    """
    if aggregation_modes is None:
        aggregation_modes = ["mean", "median", "max", {"mode": "top_k", "k": 100}]
    parsed_modes = _parse_aggregation_modes(aggregation_modes)

    detections = detector.detect(image)
    if not detections:
        return []

    detections = segmentor.segment_detections(image, detections)
    depth_map = depth_estimator.estimate_depth(image)

    if scale_depth_to_meters:
        depth_map, _, _ = depth_estimator.scale_depth_to_meters(depth_map)
        if closest_side == "high":
            closest_side = "low"

    H, W = image.shape[:2]
    out = []
    for d in detections:
        if not _REQUIRED_DETECTION_KEYS.issubset(d.keys()):
            continue

        values = _extract_depth_in_region(depth_map, d.get("mask"), d["bbox"])
        depth_stats = {}
        for key, spec in parsed_modes.items():
            try:
                depth_stats[key] = _aggregate_values(
                    values, mode=spec["mode"], k=spec["k"], p=spec["p"],
                    closest_side=closest_side,
                )
            except Exception:
                depth_stats[key] = float("nan")

        cx_norm = _compute_centroid_from_mask_or_bbox(d.get("mask"), d["bbox"], W)
        direction, angle_deg = _compute_direction_and_angle(cx_norm)

        out.append({
            **d,
            "depth_stats": depth_stats,
            "direction": direction,
            "angle_deg": angle_deg,
            "centroid_x_norm": cx_norm,
        })
    return out
