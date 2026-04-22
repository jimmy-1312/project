"""
Scene Analyzer: Unified YOLO + MobileSAM + Depth Integration

Integrates YOLODetector, MobileSAMSegmentor, and DepthEstimator to produce
rich per-object scene analysis: depth aggregation, direction, angle.

Public API:
  - analyze_scene(): Main entry point for single-image analysis

Internal helpers:
  - _parse_aggregation_modes(): Parse user aggregation_modes specification
  - _extract_depth_in_region(): Extract depth values from mask or bbox
  - _compute_centroid_from_mask_or_bbox(): Compute object centroid
  - _compute_direction_and_angle(): Map centroid x-norm to direction/angle
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import config
from src.depth_estimator import _aggregate_values

logger = logging.getLogger(__name__)

# Required keys in each detection dict after segmentor enrichment.
_REQUIRED_DETECTION_KEYS = frozenset({
    "bbox", "mask", "class_id", "class_name", "confidence", "mask_score",
})


def _parse_aggregation_modes(
    modes: Union[List[str], List[Dict]]
) -> Dict[str, Dict]:
    """
    Parse and normalize aggregation_modes specification.

    Converts user input into a canonical dict format with validated parameters.

    Args:
        modes: List of mode specifications. Each item can be:
            - str: Simple mode name ("mean", "median", "max", "min")
            - dict: Complex mode with parameters
              {"mode": "top_k"|"top_p", "k": int|None, "p": float|None}

    Returns:
        Dict[mode_key, mode_spec] where:
          - mode_key: Unique key for depth_stats (e.g., "mean", "top_k_200", "top_p_0.1")
          - mode_spec: Dict with keys {mode, k, p}

    Raises:
        TypeError: If modes is not a list/tuple.
        ValueError: If mode item is not str or dict, or has invalid syntax.

    Examples:
        >>> _parse_aggregation_modes(["mean", "median"])
        {
            "mean": {"mode": "mean", "k": None, "p": None},
            "median": {"mode": "median", "k": None, "p": None},
        }

        >>> _parse_aggregation_modes([{"mode": "top_k", "k": 50}])
        {
            "top_k_50": {"mode": "top_k", "k": 50, "p": None},
        }
    """
    if not isinstance(modes, (list, tuple)):
        raise TypeError(
            f"aggregation_modes must be a list or tuple, got {type(modes).__name__}"
        )

    result = {}
    valid_simple_modes = {"mean", "median", "max", "min"}

    for item in modes:
        if isinstance(item, str):
            # Simple string mode
            if item not in valid_simple_modes:
                raise ValueError(
                    f"Unknown aggregation mode '{item}'. "
                    f"Valid simple modes: {valid_simple_modes}"
                )
            mode_key = item
            mode_spec = {"mode": item, "k": None, "p": None}

        elif isinstance(item, dict):
            # Complex dict mode
            if "mode" not in item:
                raise ValueError(
                    "Dict mode must include 'mode' key. "
                    "Example: {'mode': 'top_k', 'k': 200}"
                )

            mode = item["mode"]
            if mode not in {"top_k", "top_p"}:
                raise ValueError(
                    f"Dict mode must be 'top_k' or 'top_p', got '{mode}'"
                )

            if mode == "top_k":
                k = item.get("k")
                if k is None:
                    raise ValueError(
                        "top_k mode requires 'k' parameter. "
                        "Example: {'mode': 'top_k', 'k': 200}"
                    )
                if not isinstance(k, int) or k <= 0:
                    raise ValueError(f"top_k 'k' must be positive int, got {k}")
                mode_key = f"top_k_{k}"
                mode_spec = {"mode": "top_k", "k": k, "p": None}

            else:  # top_p
                p = item.get("p")
                if p is None:
                    raise ValueError(
                        "top_p mode requires 'p' parameter. "
                        "Example: {'mode': 'top_p', 'p': 0.1}"
                    )
                if not isinstance(p, (int, float)) or not (0.0 < p <= 1.0):
                    raise ValueError(f"top_p 'p' must be in (0, 1], got {p}")
                # Format p nicely: 0.1 -> "0.1", 0.01 -> "0.01"
                mode_key = f"top_p_{p:.2g}"
                mode_spec = {"mode": "top_p", "k": None, "p": float(p)}

        else:
            raise ValueError(
                f"aggregation_modes item must be str or dict, got {type(item).__name__}. "
                "Example: ['mean', 'median', {'mode': 'top_k', 'k': 50}]"
            )

        result[mode_key] = mode_spec

    return result


def _extract_depth_in_region(
    depth_map: np.ndarray,
    mask: Optional[np.ndarray],
    bbox: np.ndarray,
) -> np.ndarray:
    """
    Extract depth values from a region (mask or bbox).

    Mask is prioritized; falls back to bbox if mask is None or empty.

    Args:
        depth_map: (H, W) float32 depth map
        mask: (H, W) bool or None. If provided and non-empty, use mask.
        bbox: [x1, y1, x2, y2] fallback region

    Returns:
        1D array of depth values (size >= 0). May be empty if no valid pixels.
    """
    # Try mask first
    if mask is not None and np.any(mask):
        depth_values = depth_map[np.asarray(mask) > 0]
        if len(depth_values) > 0:
            return depth_values

    # Fallback to bbox
    H, W = depth_map.shape
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(round(float(x1))))
    y1 = max(0, int(round(float(y1))))
    x2 = min(W, int(round(float(x2))))
    y2 = min(H, int(round(float(y2))))

    if x2 <= x1 or y2 <= y1:
        return np.array([], dtype=np.float32)

    region = depth_map[y1:y2, x1:x2]
    return region.ravel()


def _compute_centroid_from_mask_or_bbox(
    mask: Optional[np.ndarray],
    bbox: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute centroid (center of mass) from mask or bbox.

    Mask is prioritized; falls back to bbox center if mask is None or empty.

    Args:
        mask: (H, W) bool or None
        bbox: [x1, y1, x2, y2]

    Returns:
        (centroid_x, centroid_y) in pixel coordinates
    """
    if mask is not None and np.any(mask):
        y_indices, x_indices = np.where(np.asarray(mask) > 0)
        if len(x_indices) > 0:
            return float(np.mean(x_indices)), float(np.mean(y_indices))

    # Fallback to bbox center
    x1, y1, x2, y2 = bbox
    centroid_x = (float(x1) + float(x2)) / 2.0
    centroid_y = (float(y1) + float(y2)) / 2.0
    return centroid_x, centroid_y


def _aggregate_for_mode(
    depth_values: np.ndarray,
    mode_spec: Dict,
    closest_side: str,
) -> float:
    """
    Aggregate depth values for a parsed mode spec.

    Delegates to depth_estimator._aggregate_values (NaN/Inf filtering,
    empty-array handling, top-k/top-p selection are all handled there).

    Args:
        depth_values: 1D array of depth samples (may be empty).
        mode_spec: Parsed mode dict with {mode, k, p}.
        closest_side: "high" (larger = closer) or "low" (smaller = closer).

    Returns:
        Aggregated depth value (float), NaN if no valid samples.
    """
    return _aggregate_values(
        depth_values,
        mode=mode_spec["mode"],
        k=mode_spec.get("k"),
        p=mode_spec.get("p"),
        closest_side=closest_side,
    )


def _compute_direction_and_angle(
    centroid_x_norm: float,
) -> Tuple[str, float]:
    """
    Compute direction and angle from normalized x coordinate.

    Args:
        centroid_x_norm: Normalized x [0, 1], where 0.5 = center

    Returns:
        (direction, angle_deg):
          direction in ["left", "center", "right", "unknown"]
          angle_deg in [-FOV/2, +FOV/2]
    """
    fov = getattr(config, "HORIZONTAL_FOV", 60.0)
    dir_left = getattr(config, "DIR_LEFT", 0.33)
    dir_right = getattr(config, "DIR_RIGHT", 0.67)

    # Compute angle: -FOV/2 (left) to +FOV/2 (right)
    angle_deg = (centroid_x_norm - 0.5) * fov

    # Classify direction
    if centroid_x_norm < dir_left:
        direction = "left"
    elif centroid_x_norm > dir_right:
        direction = "right"
    else:
        direction = "center"

    return direction, float(angle_deg)


def analyze_scene(
    image: np.ndarray,
    detector,
    segmentor,
    depth_estimator,
    aggregation_modes: Union[List[str], List[Dict]],
    scale_depth_to_meters: bool = False,
    closest_side: str = "high",
) -> List[Dict]:
    """
    Analyze a single image for objects with depth and direction information.

    Entry point for per-image scene analysis. Combines YOLO detection,
    MobileSAM segmentation, and Depth Anything V2 estimation to produce
    rich per-object results including depth aggregation, direction,
    and angle measurements.

    Args:
        image (np.ndarray): Input image, shape (H, W, 3), dtype uint8, RGB.
        detector: YOLODetector instance with detect(image) -> List[Dict].
        segmentor: MobileSAMSegmentor instance with
            segment_detections(image, detections) -> detections (with mask, mask_score).
        depth_estimator: DepthEstimator instance with
            estimate_depth(image) -> (H, W) float32.
        aggregation_modes (Union[List[str], List[Dict]]): Depth aggregation modes.
            Examples:
              ["mean", "median", {"mode": "top_k", "k": 200}, {"mode": "top_p", "p": 0.1}]
        scale_depth_to_meters (bool): If True, call depth_estimator.scale_depth_to_meters()
            to convert relative depth to metric. Default False.
        closest_side (str): "high" (large depth = close) or "low" (small = close).
            Default "high" (Depth Anything V2 convention).

    Returns:
        List[Dict]: Per-object results. Each dict contains:
            {
                "class_id": int,
                "class_name": str,
                "confidence": float,
                "bbox": np.ndarray [x1, y1, x2, y2],
                "mask": np.ndarray (H, W) bool,
                "mask_score": float,
                "depth_stats": {mode_key: aggregated_value, ...},
                "direction": "left"|"center"|"right"|"unknown",
                "angle_deg": float,
                "centroid_x_norm": float,
            }

    Raises:
        TypeError: If image is not ndarray or wrong shape.
        ValueError: If image size is 0, channels != 3, or aggregation_modes syntax error.
        RuntimeError: If detector, segmentor, or depth_estimator fails.

    Notes:
        - Empty detection (len=0) returns [].
        - Empty mask falls back to bbox for depth aggregation.
        - NaN depth values are auto-filtered in aggregation.
        - Centroid clipped to [0, 1] in normalized x.
    """
    logger.debug("Starting scene analysis")

    # ===== Input Validation =====
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"image must be np.ndarray, got {type(image).__name__}"
        )
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"image must have shape (H, W, 3), got {image.shape}"
        )
    H, W = image.shape[:2]
    if H == 0 or W == 0:
        raise ValueError(
            f"image dimensions must be > 0, got ({H}, {W})"
        )

    # Parse aggregation modes
    try:
        parsed_modes = _parse_aggregation_modes(aggregation_modes)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Invalid aggregation_modes: {e}"
        ) from e

    logger.debug(f"Image shape: {image.shape}, modes: {list(parsed_modes.keys())}")

    # ===== YOLO Detection =====
    try:
        detections = detector.detect(image)
    except Exception as e:
        raise RuntimeError(
            f"YOLODetector.detect() failed: {e}"
        ) from e

    if not detections:
        logger.info("No detections found")
        return []

    logger.info(f"Detected {len(detections)} objects")

    # ===== Depth Estimation =====
    try:
        depth_map = depth_estimator.estimate_depth(image)
    except Exception as e:
        raise RuntimeError(
            f"DepthEstimator.estimate_depth() failed: {e}"
        ) from e

    if depth_map.ndim != 2:
        raise ValueError(
            f"depth_map must be 2D, got shape {depth_map.shape}"
        )

    # ===== MobileSAM Segmentation =====
    try:
        detections = segmentor.segment_detections(image, detections)
    except Exception as e:
        raise RuntimeError(
            f"MobileSAMSegmentor.segment_detections() failed: {e}"
        ) from e

    # ===== Depth Scaling (optional) =====
    if scale_depth_to_meters:
        try:
            scaled_depth, _, _ = depth_estimator.scale_depth_to_meters(depth_map)
            depth_map = scaled_depth
            # Metric depth inverts the "closer = larger" convention. If the
            # caller left the default closest_side="high", flip it so that
            # top_k/top_p still select the *closest* pixels.
            if closest_side == "high":
                closest_side = "low"
                logger.debug("Scaled to metric: flipped closest_side to 'low'")
            else:
                logger.debug("Scaled depth to metric")
        except Exception as e:
            raise RuntimeError(
                f"DepthEstimator.scale_depth_to_meters() failed: {e}"
            ) from e

    # ===== Per-Object Enrichment =====
    results = []
    for i, detection in enumerate(detections):
        logger.debug(f"Processing detection {i+1}/{len(detections)}")

        # Validate detection structure
        missing = _REQUIRED_DETECTION_KEYS - detection.keys()
        if missing:
            raise RuntimeError(f"Detection {i} missing keys: {set(missing)}")

        bbox = detection["bbox"]
        mask = detection["mask"]
        class_id = detection["class_id"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]
        mask_score = detection["mask_score"]

        # Extract depth in region and aggregate per mode
        depth_values = _extract_depth_in_region(depth_map, mask, bbox)
        depth_stats = {
            mode_key: _aggregate_for_mode(depth_values, mode_spec, closest_side)
            for mode_key, mode_spec in parsed_modes.items()
        }

        # Compute centroid and direction
        centroid_x, centroid_y = _compute_centroid_from_mask_or_bbox(mask, bbox)
        centroid_x_norm = np.clip(centroid_x / max(W, 1), 0.0, 1.0)
        direction, angle_deg = _compute_direction_and_angle(centroid_x_norm)

        # Build result
        result = {
            "class_id": int(class_id),
            "class_name": str(class_name),
            "confidence": float(confidence),
            "bbox": np.asarray(bbox, dtype=np.float32),
            "mask": np.asarray(mask, dtype=bool),
            "mask_score": float(mask_score),
            "depth_stats": depth_stats,
            "direction": direction,
            "angle_deg": angle_deg,
            "centroid_x_norm": float(centroid_x_norm),
        }
        results.append(result)

    logger.info(f"Scene analysis complete: {len(results)} objects analyzed")
    return results
