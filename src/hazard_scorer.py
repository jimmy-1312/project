"""
Hazard Scorer
=============

Ranks objects detected by ``analyze_scene()`` (scene_analyzer.py) so the
visual assistant announces the most urgent ones first.

Input contract
--------------
Each item in the list returned by ``analyze_scene()`` contains::

    {
        "class_name":      str,
        "confidence":      float,
        "bbox":            np.ndarray [x1, y1, x2, y2],
        "mask":            np.ndarray (H, W) bool,
        "mask_score":      float,
        "depth_stats":     {mode_key: float, ...},   # e.g. {"mean": 0.7, "top_k_100": 0.85}
        "direction":       str,
        "angle_deg":       float,
        "centroid_x_norm": float,
    }

In Depth Anything V2's relative-depth convention (after min-max normalisation
in DepthEstimator.estimate_depth), **LARGER** depth value means **CLOSER** to
the camera.  The scorer therefore treats larger depth_stats values as higher
proximity.  If you call scale_depth_to_meters() first (metric depth, smaller =
closer), pass ``closest_side="low"`` to the scorer.

Scoring formula
---------------
score = w_danger * danger + w_proximity * proximity + w_size * size

All three sub-scores are in [0, 1].  The weights default to::

    danger    = 0.50
    proximity = 0.35
    size      = 0.15

YOLO confidence acts as a mild multiplier:
    final = raw_score × (0.5 + 0.5 × confidence)

Usage
-----
    from src.hazard_scorer import HazardScorer

    scorer = HazardScorer()
    ranked = scorer.rank(scene_results, image_h=480, image_w=640)

    for obj in ranked:
        print(obj["class_name"], obj["priority_score"])
        print(obj["score_breakdown"])
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Danger look-up table
# ──────────────────────────────────────────────────────────────────────────────
# Scale: 0.0 = harmless clutter, 1.0 = immediate life threat.
# Keys are lower-cased class names so they survive minor label variations.
# New fine-tuned classes (★) are pre-populated ready for when YOLO is updated.

DANGER_TABLE: Dict[str, float] = {
    # ── Fine-tuned additions ★ ───────────────────────────────────────────────
    "fire":              1.00,
    "smoke":             0.95,
    "water_on_ground":   0.90,   # slip / fall hazard
    "wet_floor":         0.88,
    "knife":             0.90,
    "scissors":          0.80,

    # ── Moving / collision hazards (COCO-80) ────────────────────────────────
    "person":            0.70,   # people move unpredictably
    "dog":               0.60,
    "cat":               0.35,
    "bicycle":           0.65,
    "motorcycle":        0.75,
    "car":               0.80,
    "bus":               0.85,
    "truck":             0.85,
    "horse":             0.70,
    "cow":               0.65,
    "bear":              0.90,
    "elephant":          0.80,
    "zebra":             0.60,
    "giraffe":           0.55,

    # ── Hot / sharp objects ──────────────────────────────────────────────────
    "oven":              0.65,
    "toaster":           0.50,
    "microwave":         0.40,
    "fork":              0.45,
    "wine glass":        0.40,   # shatters
    "vase":              0.35,   # shatters

    # ── Tripping / bumping hazards ───────────────────────────────────────────
    "chair":             0.55,
    "bench":             0.45,
    "dining table":      0.50,
    "table":             0.50,
    "couch":             0.40,
    "bed":               0.35,
    "toilet":            0.40,
    "suitcase":          0.55,
    "backpack":          0.45,
    "handbag":           0.35,
    "sports ball":       0.50,   # rolls — high trip risk
    "skateboard":        0.65,
    "bottle":            0.30,

    # ── Household items (lower hazard) ───────────────────────────────────────
    "sink":              0.25,
    "refrigerator":      0.30,
    "tv":                0.30,
    "laptop":            0.25,
    "cup":               0.25,
    "bowl":              0.20,
    "cell phone":        0.10,
    "book":              0.10,
    "clock":             0.10,
    "potted plant":      0.20,
    "umbrella":          0.30,
    "keyboard":          0.10,
    "mouse":             0.10,

    # ── Default ──────────────────────────────────────────────────────────────
    "__default__":       0.20,
}

# Preferred depth_stats keys in order of reliability.
# The scorer picks the first key that exists and is finite.
_DEPTH_KEY_PREFERENCE = [
    "top_k_100", "top_k_50", "top_p_0.2", "top_p_0.1",
    "max", "median", "mean",
]

# Proximity decay constant k in  score = exp(-k * depth_norm).
# With k=3: depth_norm=0 → 1.0, depth_norm=0.5 → ~0.22, depth_norm=1.0 → ~0.05
_PROXIMITY_DECAY: float = 3.0


class HazardScorer:
    """
    Scores and sorts ``analyze_scene()`` results by urgency.

    Higher ``priority_score`` → announce to the user sooner.

    Parameters
    ----------
    w_danger : float
        Weight for object-class danger. Default 0.50.
    w_proximity : float
        Weight for how close the object is. Default 0.35.
    w_size : float
        Weight for mask area (fraction of image). Default 0.15.
    proximity_decay : float
        Controls steepness of the proximity exponential. Default 3.0.
    closest_side : str
        ``"high"`` if larger depth value = closer (Depth Anything V2
        relative depth, the default).  ``"low"`` if smaller = closer
        (metric / scaled depth).
    danger_table : dict, optional
        Override the built-in danger look-up table.
    """

    def __init__(
        self,
        w_danger: float = 0.50,
        w_proximity: float = 0.35,
        w_size: float = 0.15,
        proximity_decay: float = _PROXIMITY_DECAY,
        closest_side: str = "high",
        danger_table: Optional[Dict[str, float]] = None,
    ) -> None:
        if any(w < 0 for w in (w_danger, w_proximity, w_size)):
            raise ValueError("All weights must be non-negative.")
        total = w_danger + w_proximity + w_size
        if total == 0:
            raise ValueError("Weights must not all be zero.")

        self.w_danger = w_danger / total
        self.w_proximity = w_proximity / total
        self.w_size = w_size / total
        self.proximity_decay = proximity_decay
        self.closest_side = closest_side
        self.danger_table = danger_table or DANGER_TABLE

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def score(
        self,
        obj: Dict,
        image_h: int,
        image_w: int,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute the priority score for a single scene-analysis result.

        Parameters
        ----------
        obj : dict
            One element from ``analyze_scene()`` output.
        image_h, image_w : int
            Original image dimensions used to normalise mask size.

        Returns
        -------
        final_score : float
            Weighted priority score in [0, 1].
        breakdown : dict
            Component scores for logging / debugging:
            ``{"danger", "proximity", "size", "depth_used", "final"}``.
        """
        class_name = str(obj.get("class_name", "")).lower()
        confidence = float(obj.get("confidence", 1.0))
        depth_stats: Dict[str, float] = obj.get("depth_stats") or {}
        mask = obj.get("mask")

        danger    = self._danger_score(class_name)
        depth_val, depth_key = self._pick_depth(depth_stats)
        proximity = self._proximity_score(depth_val)
        size      = self._size_score(mask, image_h, image_w)

        raw = (
            self.w_danger    * danger
          + self.w_proximity * proximity
          + self.w_size      * size
        )
        # Confidence multiplier keeps uncertain detections from dominating.
        final = raw * (0.5 + 0.5 * confidence)

        breakdown = {
            "danger":     round(danger,    4),
            "proximity":  round(proximity, 4),
            "size":       round(size,      4),
            "depth_used": depth_key,
            "depth_val":  round(float(depth_val), 4) if np.isfinite(depth_val) else None,
            "final":      round(final,     4),
        }
        return round(final, 4), breakdown

    def rank(
        self,
        objects: List[Dict],
        image_h: int,
        image_w: int,
    ) -> List[Dict]:
        """
        Score every object and return them sorted highest priority first.

        Each dict gets two new keys added:
          ``priority_score`` (float) and ``score_breakdown`` (dict).

        Parameters
        ----------
        objects : list of dict
            Output of ``analyze_scene()``.
        image_h, image_w : int
            Image dimensions.

        Returns
        -------
        list of dict
            Same objects, sorted by ``priority_score`` descending.
        """
        scored = []
        for obj in objects:
            final, breakdown = self.score(obj, image_h, image_w)
            updated = dict(obj)
            updated["priority_score"]  = final
            updated["score_breakdown"] = breakdown
            scored.append(updated)

        scored.sort(key=lambda d: d["priority_score"], reverse=True)
        logger.debug(f"Ranked {len(scored)} objects")
        return scored

    def top_n(
        self,
        objects: List[Dict],
        image_h: int,
        image_w: int,
        n: int,
    ) -> List[Dict]:
        """Return only the top-n most urgent objects."""
        return self.rank(objects, image_h, image_w)[:n]

    def announcement_lines(
        self,
        objects: List[Dict],
        image_h: int,
        image_w: int,
        n: Optional[int] = None,
    ) -> List[str]:
        """
        Return short human-readable strings in priority order.

        Useful for feeding directly into the LLM or for quick debugging.

        Example output::

            ["knife  score=0.91  depth=0.87  → LEFT",
             "person score=0.74  depth=0.61  → CENTER"]
        """
        ranked = self.rank(objects, image_h, image_w)
        if n is not None:
            ranked = ranked[:n]

        lines = []
        for obj in ranked:
            name      = obj.get("class_name", "unknown")
            score     = obj.get("priority_score", 0.0)
            direction = obj.get("direction", "?").upper()
            bd        = obj.get("score_breakdown", {})
            depth_val = bd.get("depth_val")
            depth_str = f"{depth_val:.3f}" if depth_val is not None else "nan"
            lines.append(
                f"{name:<20s}  score={score:.3f}  depth={depth_str}  → {direction}"
            )
        return lines

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _danger_score(self, class_name: str) -> float:
        """Look up danger from the table; fall back to default."""
        return float(
            self.danger_table.get(
                class_name,
                self.danger_table.get("__default__", 0.20),
            )
        )

    def _pick_depth(self, depth_stats: Dict[str, float]) -> Tuple[float, str]:
        """
        Choose the best available depth value from depth_stats.

        Uses ``_DEPTH_KEY_PREFERENCE`` order so the most robust estimate
        (top-k closest pixels) is preferred over a plain mean.

        Returns
        -------
        (value, key) — value is NaN if nothing usable is found.
        """
        for key in _DEPTH_KEY_PREFERENCE:
            val = depth_stats.get(key)
            if val is not None and np.isfinite(val):
                return float(val), key

        # Fall back to whatever is in the dict
        for key, val in depth_stats.items():
            if val is not None and np.isfinite(val):
                return float(val), key

        return float("nan"), "none"

    def _proximity_score(self, depth_val: float) -> float:
        """
        Convert a raw depth value to a proximity score in [0, 1].

        Depth Anything V2 relative depth (closest_side="high"):
            larger value → object is closer → higher urgency

        Metric depth (closest_side="low"):
            smaller value → object is closer → higher urgency

        The value is normalised to [0, 1] before the exponential so the
        decay constant is independent of the depth scale.
        """
        if not np.isfinite(depth_val):
            return 0.0   # unknown distance → conservative (not urgent)

        max_d = float(getattr(config, "MAX_DEPTH_M", 1.0))

        if self.closest_side == "high":
            # depth_val already in [0, 1] after DepthEstimator normalisation.
            # Larger → closer → proximity_norm near 1.
            proximity_norm = float(np.clip(depth_val, 0.0, 1.0))
        else:
            # Metric depth: clip to max, then invert.
            clamped = float(np.clip(depth_val, 0.0, max_d))
            proximity_norm = 1.0 - clamped / max_d

        return float(np.exp(-self.proximity_decay * (1.0 - proximity_norm)))

    def _size_score(
        self,
        mask: Optional[np.ndarray],
        image_h: int,
        image_w: int,
    ) -> float:
        """Fraction of image pixels covered by the mask."""
        if mask is None:
            return 0.0
        total = image_h * image_w
        if total == 0:
            return 0.0
        return float(np.asarray(mask).sum()) / float(total)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────────────────────────────────────

def rank_by_hazard(
    objects: List[Dict],
    image_h: int,
    image_w: int,
    top_n: Optional[int] = None,
    w_danger: float = 0.50,
    w_proximity: float = 0.35,
    w_size: float = 0.15,
    closest_side: str = "high",
) -> List[Dict]:
    """
    One-shot ranking of ``analyze_scene()`` results by hazard urgency.

    Parameters
    ----------
    objects : list of dict
        Output of ``src.scene_analyzer.analyze_scene()``.
    image_h, image_w : int
        Image dimensions for mask-size normalisation.
    top_n : int, optional
        Return only the top-n most urgent objects.  None = return all.
    w_danger, w_proximity, w_size : float
        Component weights (normalised internally).
    closest_side : str
        ``"high"`` for relative depth (default), ``"low"`` for metric depth.

    Returns
    -------
    list of dict
        Sorted objects with ``priority_score`` and ``score_breakdown`` added.

    Example
    -------
    >>> from src.scene_analyzer import analyze_scene
    >>> from src.hazard_scorer import rank_by_hazard
    >>>
    >>> results = analyze_scene(image_rgb, detector, segmentor, depth_estimator,
    ...                         aggregation_modes=["mean", {"mode": "top_k", "k": 100}])
    >>> ranked = rank_by_hazard(results, image_h=480, image_w=640, top_n=3)
    >>> for obj in ranked:
    ...     print(obj["class_name"], obj["priority_score"])
    """
    scorer = HazardScorer(
        w_danger=w_danger,
        w_proximity=w_proximity,
        w_size=w_size,
        closest_side=closest_side,
    )
    ranked = scorer.rank(objects, image_h, image_w)
    return ranked[:top_n] if top_n is not None else ranked
