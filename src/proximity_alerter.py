"""
Proximity-based Object Alert Generation

Given scene analysis results, rank nearby objects by proximity (distance)
and generate natural language alerts for visually impaired users.

Public API:
  - detect_by_proximity()    # Main function: rank by distance, generate alerts
  - format_nearest_alert()   # Format single detection as natural language
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _extract_distance_m(depth_stats: Optional[Dict]) -> Optional[float]:
    """
    Extract a single distance estimate (meters) from depth_stats.

    Preference order: top_k_100 (closest-K mean, robust) > mean > median.
    Returns None if no finite, positive depth is available.

    Args:
        depth_stats: Dict from detection["depth_stats"] or None.

    Returns:
        Distance in meters (float) or None if unavailable/invalid.
    """
    if not depth_stats or not isinstance(depth_stats, dict):
        return None

    # Preference order: prioritize top_k_100 (closest pixels, most robust)
    for key in ("top_k_100", "mean", "median"):
        val = depth_stats.get(key)
        if val is None:
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        # Only accept finite, positive values (distance must be > 0)
        if np.isfinite(f) and f > 0:
            return f

    return None


_DIRECTION_WORD = {
    "left": "left",
    "center": "ahead",
    "right": "right",
}


def _direction_word(direction: Optional[str]) -> str:
    """
    Map a direction key to its English natural-language word.

    Returns "" for "unknown" / None / unrecognized values so the caller
    can drop direction from the alert string.
    """
    if direction in (None, "unknown"):
        return ""
    return _DIRECTION_WORD.get(direction, "")


def format_nearest_alert(detection: Dict) -> str:
    """
    Format a single detection dict as a natural-language alert string.

    Generates a short, actionable alert sentence suitable for audio playback.
    Example outputs:
        "left 1.2m person"
        "ahead 0.5m chair"
        "3.0m window"            # direction unavailable
        "left person"            # distance unavailable

    Args:
        detection: Detection dict with keys:
            - "class_name" (required): Object class name
            - "distance_m" (optional): Distance in meters
            - "direction" (optional): One of "left", "center", "right", "unknown"

    Returns:
        Natural-language alert string.
    """
    class_name = detection.get("class_name", "object")
    distance_m = detection.get("distance_m")
    direction = detection.get("direction")

    # Format distance
    distance_str = ""
    if distance_m is not None:
        try:
            dist_float = float(distance_m)
            if np.isfinite(dist_float) and dist_float > 0:
                distance_str = f"{dist_float:.1f}m"
        except (TypeError, ValueError):
            pass

    direction_str = _direction_word(direction)

    # English format: "{direction} {distance}m {class_name}"
    parts = []
    if direction_str:
        parts.append(direction_str)
    if distance_str:
        parts.append(distance_str)
    parts.append(class_name)
    return " ".join(parts)


def detect_by_proximity(
    detections: List[Dict],
    top_k: int = 5,
) -> List[Dict]:
    """
    Rank detections by proximity (distance) and return top-K with alerts.

    Main entry point for proximity-based ranking. Takes the raw output of
    analyze_scene() and:
        1. Extracts distance from each detection's depth_stats
        2. Filters out detections without valid distance
        3. Sorts by distance (closest first)
        4. Selects top-K
        5. Generates a natural-language alert for each

    Args:
        detections: List of detection dicts from analyze_scene().
            Each must have "class_name" and optionally "depth_stats",
            "direction", "angle_deg".
        top_k: Maximum number of objects to return.
            - top_k > 0: at most top_k closest objects
            - top_k <= 0: no limit (returns all sorted by distance)

    Returns:
        List of dicts sorted by distance ascending (closest first):
            [
                {
                    "class_name": str,
                    "direction": str,
                    "distance_m": float,
                    "angle_deg": float,
                    "alert": str,    # natural-language alert sentence
                },
                ...
            ]

    Raises:
        TypeError: If `detections` is not a list.

    Notes:
        - Detections without valid distance (NaN, None, non-positive) are
          excluded — alerts without distance are not actionable.
        - Empty input list returns an empty list.
        - Top-K limit is applied AFTER distance validity filtering.
    """
    if not isinstance(detections, list):
        raise TypeError(
            f"detections must be a list, got {type(detections).__name__}"
        )

    # Extract and validate distance per detection
    validated = []
    for detection in detections:
        depth_stats = detection.get("depth_stats") if isinstance(detection, dict) else None
        distance = _extract_distance_m(depth_stats)

        if distance is not None and np.isfinite(distance) and distance > 0:
            validated.append((detection, distance))

    # Sort by distance ascending (closest first)
    validated.sort(key=lambda x: x[1])

    # Apply top-K limit
    if top_k > 0:
        selected = validated[:top_k]
    else:
        selected = validated

    # Build result with alerts
    result = []
    for detection, distance in selected:
        alert_input = dict(detection)
        alert_input["distance_m"] = distance
        alert_text = format_nearest_alert(alert_input)

        result.append({
            "class_name": str(detection.get("class_name", "object")),
            "direction": detection.get("direction", "unknown"),
            "distance_m": float(distance),
            "angle_deg": float(detection.get("angle_deg", 0.0)),
            "alert": alert_text,
        })

    return result
