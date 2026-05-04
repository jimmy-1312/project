"""
Proximity Alerter
=================

Implements the revised pipeline proposed in the group chat (May 3, 2026)::

    Depth Anything V2 → depth map
        ↓
    ProximityAlerter.find_nearby_regions()   ← this module
        ↓  crops of near regions
    YOLO detector  (runs only on nearby crops → fewer false positives)
        ↓
    MobileSAM segmentation
        ↓
    HazardScorer

Why this order?
---------------
Running YOLO on the full image when the model has hundreds of classes
produces many detections of distant objects that are not actionable for a
blind user navigating a room.  By first asking "what is *close* to the
camera?" we can focus YOLO's attention on the regions that matter, reducing
noise and (when YOLO is extended to hundreds of classes) keeping
inference feasible.

Depth Anything V2 convention
-----------------------------
After ``DepthEstimator.estimate_depth()`` the depth map is normalised to
[0, 1] with **LARGER** value = **CLOSER** to the camera.  All thresholds
in this module use this convention by default.  Pass
``closest_side="low"`` when working with metric (scaled) depth where smaller
values are closer.

Public API
----------
``find_nearby_regions(depth_map, threshold, ...)``
    Return bounding boxes of regions above the threshold.

``ProximityAlerter`` class
    Stateful version with configurable parameters and helpers for
    integrating with the rest of the pipeline.

``alert_on_image(image, depth_map, detector, ...)``
    End-to-end convenience function: find close regions → run YOLO on
    crops → return detections with full-image coordinates restored.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────────────────────────────────────

def _binary_near_mask(
    depth_map: np.ndarray,
    threshold: float,
    closest_side: str,
) -> np.ndarray:
    """
    Return a binary mask of pixels that are *close* to the camera.

    Parameters
    ----------
    depth_map : (H, W) float32
    threshold : float
        Pixels on the "close" side of this value are marked True.
    closest_side : str
        ``"high"`` → pixels with depth > threshold are close (relative depth).
        ``"low"``  → pixels with depth < threshold are close (metric depth).

    Returns
    -------
    (H, W) bool
    """
    if closest_side == "high":
        return depth_map > threshold
    return depth_map < threshold


def _connected_component_boxes(
    binary_mask: np.ndarray,
    min_area_fraction: float,
    padding: int,
    image_shape: Tuple[int, int],
) -> List[np.ndarray]:
    """
    Find connected components in a binary mask and return their bounding boxes.

    Parameters
    ----------
    binary_mask : (H, W) bool
    min_area_fraction : float
        Components smaller than ``min_area_fraction * H * W`` are discarded.
    padding : int
        Extra pixels added around each box (clamped to image bounds).
    image_shape : (H, W)

    Returns
    -------
    List of np.ndarray, each [x1, y1, x2, y2] dtype float32.
    """
    H, W = image_shape
    min_area = max(1, int(min_area_fraction * H * W))

    mask_u8 = binary_mask.astype(np.uint8) * 255
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    boxes = []
    for label in range(1, n_labels):   # skip background (label 0)
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(W, x + w + padding)
        y2 = min(H, y + h + padding)

        boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))

    return boxes


def _restore_bbox_coordinates(
    crop_detections: List[Dict],
    crop_box: np.ndarray,
) -> List[Dict]:
    """
    Translate bounding boxes from crop-local to full-image coordinates.

    Parameters
    ----------
    crop_detections : list of dict
        Detections whose ``bbox`` is in crop-local pixel coords.
    crop_box : np.ndarray [x1, y1, x2, y2]
        The crop's position in the full image.

    Returns
    -------
    Same dicts with ``bbox`` adjusted to full-image coordinates and
    ``crop_box`` added for provenance.
    """
    x_offset = float(crop_box[0])
    y_offset = float(crop_box[1])

    restored = []
    for det in crop_detections:
        updated = dict(det)
        bbox = np.asarray(det["bbox"], dtype=np.float32)
        updated["bbox"] = np.array(
            [
                bbox[0] + x_offset,
                bbox[1] + y_offset,
                bbox[2] + x_offset,
                bbox[3] + y_offset,
            ],
            dtype=np.float32,
        )
        updated["crop_box"] = crop_box.copy()
        restored.append(updated)
    return restored


# ──────────────────────────────────────────────────────────────────────────────
# Core function
# ──────────────────────────────────────────────────────────────────────────────

def find_nearby_regions(
    depth_map: np.ndarray,
    threshold: float = 0.6,
    closest_side: str = "high",
    min_area_fraction: float = 0.005,
    padding: int = 20,
    max_regions: int = 10,
) -> List[np.ndarray]:
    """
    Find bounding boxes of regions that are close to the camera.

    Parameters
    ----------
    depth_map : np.ndarray
        (H, W) float, output of ``DepthEstimator.estimate_depth()``.
    threshold : float
        Proximity threshold.  Pixels on the "close" side of this value
        are treated as nearby.

        * For relative depth (``closest_side="high"``): values in [0, 1];
          0.6 means "the closest 40 % of the depth range".
        * For metric depth (``closest_side="low"``): in metres;
          e.g. 2.0 means "everything closer than 2 m".

        Default 0.6 (relative depth convention).
    closest_side : str
        ``"high"`` (default) for relative depth; ``"low"`` for metric depth.
    min_area_fraction : float
        Ignore regions smaller than this fraction of the image.
        Default 0.005 (0.5 % of pixels, ~1500 px on a 640×480 image).
    padding : int
        Extra pixels added around each region box. Default 20.
    max_regions : int
        Return at most this many regions (largest first). Default 10.

    Returns
    -------
    list of np.ndarray
        Each element is [x1, y1, x2, y2] (float32) in pixel coordinates,
        sorted largest-area first.  Empty list if nothing is close enough.

    Raises
    ------
    ValueError
        If ``depth_map`` is not 2-D.
    """
    if depth_map.ndim != 2:
        raise ValueError(f"depth_map must be 2D, got shape {depth_map.shape}")

    H, W = depth_map.shape
    near_mask = _binary_near_mask(depth_map, threshold, closest_side)

    boxes = _connected_component_boxes(
        near_mask,
        min_area_fraction=min_area_fraction,
        padding=padding,
        image_shape=(H, W),
    )

    if not boxes:
        logger.debug("No nearby regions found above threshold %.2f", threshold)
        return []

    # Sort by box area, largest first
    boxes.sort(
        key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
        reverse=True,
    )

    logger.debug("Found %d nearby regions (threshold=%.2f)", len(boxes), threshold)
    return boxes[:max_regions]


# ──────────────────────────────────────────────────────────────────────────────
# Stateful class
# ──────────────────────────────────────────────────────────────────────────────

class ProximityAlerter:
    """
    Stateful wrapper around ``find_nearby_regions`` with pipeline integration.

    Parameters
    ----------
    threshold : float
        Proximity threshold passed to ``find_nearby_regions``. Default 0.6.
    closest_side : str
        ``"high"`` for relative depth (default); ``"low"`` for metric depth.
    min_area_fraction : float
        Minimum region size as fraction of image. Default 0.005.
    padding : int
        Pixel padding around detected regions. Default 20.
    max_regions : int
        Maximum number of nearby regions returned per image. Default 10.
    """

    def __init__(
        self,
        threshold: float = 0.6,
        closest_side: str = "high",
        min_area_fraction: float = 0.005,
        padding: int = 20,
        max_regions: int = 10,
    ) -> None:
        if not (0.0 < threshold):
            raise ValueError("threshold must be positive")
        if closest_side not in ("high", "low"):
            raise ValueError("closest_side must be 'high' or 'low'")

        self.threshold = threshold
        self.closest_side = closest_side
        self.min_area_fraction = min_area_fraction
        self.padding = padding
        self.max_regions = max_regions

    def find_regions(self, depth_map: np.ndarray) -> List[np.ndarray]:
        """
        Return bounding boxes of nearby regions for a single depth map.

        Thin wrapper around ``find_nearby_regions`` using instance parameters.
        """
        return find_nearby_regions(
            depth_map,
            threshold=self.threshold,
            closest_side=self.closest_side,
            min_area_fraction=self.min_area_fraction,
            padding=self.padding,
            max_regions=self.max_regions,
        )

    def crop_nearby(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Crop nearby regions from the image.

        Parameters
        ----------
        image : np.ndarray (H, W, 3) uint8 RGB
        depth_map : np.ndarray (H, W) float

        Returns
        -------
        List of (crop_image, crop_box) where:
            crop_image : (h, w, 3) uint8 — the cropped region
            crop_box   : np.ndarray [x1, y1, x2, y2] — position in full image
        """
        boxes = self.find_regions(depth_map)
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append((crop, box))
        return crops

    def detect_in_nearby_regions(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        detector,
    ) -> List[Dict]:
        """
        Run YOLO only on nearby regions and return full-image detections.

        This implements the depth-first pipeline discussed in the group chat:
        instead of running YOLO on the entire image (which is expensive
        when using hundreds of classes), we first isolate the regions
        that are close and only run YOLO there.

        Parameters
        ----------
        image : np.ndarray (H, W, 3) uint8 RGB
        depth_map : np.ndarray (H, W) float
        detector : YOLODetector
            Must have a ``.detect(image_rgb)`` method returning
            list of dicts with at least ``bbox``.

        Returns
        -------
        List of detection dicts with ``bbox`` in full-image coordinates.
        All detections also carry ``crop_box`` (the nearby-region box they
        came from) for traceability.
        """
        crops = self.crop_nearby(image, depth_map)

        if not crops:
            logger.info("No nearby regions to run YOLO on.")
            return []

        all_detections: List[Dict] = []
        for crop_image, crop_box in crops:
            try:
                crop_dets = detector.detect(crop_image)
            except Exception as exc:
                logger.warning("YOLO failed on crop %s: %s", crop_box.tolist(), exc)
                continue

            # Translate bounding boxes back to full-image coordinates
            full_dets = _restore_bbox_coordinates(crop_dets, crop_box)
            all_detections.extend(full_dets)

        logger.info(
            "detect_in_nearby_regions: %d crops → %d detections",
            len(crops), len(all_detections),
        )
        return all_detections

    def visualize_regions(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw nearby-region bounding boxes on the image for debugging.

        Parameters
        ----------
        image : (H, W, 3) uint8 RGB
        depth_map : (H, W) float
        color : (R, G, B) box colour
        thickness : line thickness in pixels

        Returns
        -------
        (H, W, 3) uint8 RGB annotated copy
        """
        boxes = self.find_regions(depth_map)
        out = image.copy()
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        bgr_color = (color[2], color[1], color[0])   # OpenCV is BGR

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(out_bgr, (x1, y1), (x2, y2), bgr_color, thickness)
            cv2.putText(
                out_bgr,
                f"near[{i}]",
                (x1, max(y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                bgr_color,
                1,
            )

        return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end convenience function
# ──────────────────────────────────────────────────────────────────────────────

def alert_on_image(
    image: np.ndarray,
    depth_map: np.ndarray,
    detector,
    threshold: float = 0.6,
    closest_side: str = "high",
    min_area_fraction: float = 0.005,
    padding: int = 20,
    max_regions: int = 10,
) -> List[Dict]:
    """
    Find nearby regions in the depth map and run YOLO only on those crops.

    Convenience wrapper for one-off use without instantiating
    ``ProximityAlerter`` explicitly.

    Parameters
    ----------
    image : np.ndarray (H, W, 3) uint8 RGB
    depth_map : np.ndarray (H, W) float
        Output of ``DepthEstimator.estimate_depth()``.
    detector : YOLODetector
    threshold : float
        Proximity threshold (default 0.6 for relative depth).
    closest_side : str
        ``"high"`` for relative depth (default); ``"low"`` for metric.
    min_area_fraction : float
        Minimum region size. Default 0.005.
    padding : int
        Box padding. Default 20.
    max_regions : int
        Maximum nearby regions. Default 10.

    Returns
    -------
    List of detection dicts with full-image ``bbox`` coordinates.

    Example
    -------
    >>> from src.depth_estimator import DepthEstimator
    >>> from src.detector import YOLODetector
    >>> from src.proximity_alerter import alert_on_image
    >>>
    >>> depth_estimator = DepthEstimator()
    >>> detector = YOLODetector()
    >>>
    >>> depth_map = depth_estimator.estimate_depth(image_rgb)
    >>> detections = alert_on_image(image_rgb, depth_map, detector, threshold=0.65)
    >>> print(f"Found {len(detections)} nearby objects")
    """
    alerter = ProximityAlerter(
        threshold=threshold,
        closest_side=closest_side,
        min_area_fraction=min_area_fraction,
        padding=padding,
        max_regions=max_regions,
    )
    return alerter.detect_in_nearby_regions(image, depth_map, detector)
