"""
MobileSAM Segmentation Module
=============================

This module provides instance segmentation using MobileSAM (Mobile Segment
Anything Model). It takes bounding boxes from YOLODetector and refines them
into precise binary masks.

Two backends are supported:
  1. ``ultralytics``  — uses ``ultralytics.models.sam.Predictor`` with the
                       ``mobile_sam.pt`` weights (auto-downloaded by
                       ultralytics). Default; no extra install required.
  2. ``mobile_sam``   — uses the original ChaoningZhang/MobileSAM library
                       (``pip install git+https://github.com/ChaoningZhang/MobileSAM.git``).
                       Used as a fallback / alternative.

Both backends produce equivalent (mask, score) outputs, so the rest of the
pipeline does not care which one is in use.

Detection dict contract (matches the output of src/detector.py):
    {
        'bbox':       np.ndarray([x1, y1, x2, y2], dtype=float32),
        'confidence': float,
        'class_id':   int,
        'class_name': str,
    }

After segmentation, two extra keys are added:
    'mask':       np.ndarray (H, W) of dtype bool, True where the object is.
    'mask_score': float in [0, 1], confidence of the mask.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

import config


class MobileSAMSegmentor:
    """
    MobileSAM-based instance segmentor.

    Workflow per image::

        seg = MobileSAMSegmentor()
        detections = detector.detect(image_rgb)              # YOLODetector
        detections = seg.segment_detections(image_rgb, detections)
        # each detection now has 'mask' and 'mask_score'
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        model_type: Optional[str] = None,
        device: Optional[str] = None,
        prefer_backend: str = "ultralytics",
    ) -> None:
        self.checkpoint = checkpoint or config.MOBILE_SAM_CHECKPOINT
        self.model_type = model_type or config.MOBILE_SAM_MODEL_TYPE
        self.device = device or config.DEVICE

        self.backend: Optional[str] = None
        self.predictor = None
        self._image_set: bool = False
        self._image_shape: Tuple[int, int] = (0, 0)

        # Try the preferred backend first, then fall back to the other.
        backends_to_try = (
            ["ultralytics", "mobile_sam"]
            if prefer_backend == "ultralytics"
            else ["mobile_sam", "ultralytics"]
        )

        last_error: Optional[Exception] = None
        for backend in backends_to_try:
            try:
                if backend == "ultralytics":
                    self._init_ultralytics()
                else:
                    self._init_mobile_sam()
                self.backend = backend
                break
            except Exception as e:  # noqa: BLE001 — we want to fall through
                last_error = e
                continue

        if self.backend is None:
            raise ImportError(
                "Could not initialize MobileSAM with either the 'ultralytics' or "
                "'mobile_sam' backend. Last error: "
                f"{type(last_error).__name__}: {last_error}\n"
                "Install one of:\n"
                "  pip install ultralytics\n"
                "  pip install git+https://github.com/ChaoningZhang/MobileSAM.git"
            )

        print(
            f"MobileSAMSegmentor initialized "
            f"(backend={self.backend}, device={self.device})"
        )

    # ------------------------------------------------------------------
    # Backend initialization
    # ------------------------------------------------------------------
    def _init_ultralytics(self) -> None:
        """Initialize the ultralytics SAMPredictor with the mobile_sam.pt weights."""
        from ultralytics.models.sam import Predictor as SAMPredictor

        # If the configured checkpoint path doesn't exist on disk, fall back
        # to the bare filename — ultralytics will auto-download mobile_sam.pt
        # from its CDN into the current working directory.
        weights = self.checkpoint if os.path.isfile(self.checkpoint) else "mobile_sam.pt"

        overrides = dict(
            conf=0.25,
            task="segment",
            mode="predict",
            imgsz=1024,
            model=weights,
            device=self.device,
            verbose=False,
            save=False,
        )
        self.predictor = SAMPredictor(overrides=overrides)

    def _init_mobile_sam(self) -> None:
        """Initialize the original mobile_sam library predictor."""
        from mobile_sam import SamPredictor, sam_model_registry

        if not os.path.isfile(self.checkpoint):
            raise FileNotFoundError(
                f"MobileSAM checkpoint not found at {self.checkpoint}. "
                "Run scripts/download_models.py or download manually from "
                "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
            )

        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam.to(device=self.device)
        sam.eval()
        self.predictor = SamPredictor(sam)

    # ------------------------------------------------------------------
    # Per-image setup
    # ------------------------------------------------------------------
    def set_image(self, image_rgb: np.ndarray) -> None:
        """
        Cache the image embedding so that multiple bbox prompts on the same
        image don't re-encode it. Must be called before ``segment_box``.

        Parameters
        ----------
        image_rgb : np.ndarray, shape (H, W, 3), dtype uint8
            Image in RGB color order.
        """
        if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(
                f"Expected RGB image of shape (H, W, 3); got "
                f"{None if image_rgb is None else image_rgb.shape}"
            )

        # Both backends accept uint8 RGB ndarrays directly; they internally
        # handle resizing / preprocessing.
        self.predictor.set_image(image_rgb)
        self._image_shape = (image_rgb.shape[0], image_rgb.shape[1])
        self._image_set = True

    # ------------------------------------------------------------------
    # Single-box prompt
    # ------------------------------------------------------------------
    def segment_box(self, bbox: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Generate a binary mask for one object given its YOLO bounding box.

        Parameters
        ----------
        bbox : array-like of length 4
            [x1, y1, x2, y2] in pixel coordinates of the original image.

        Returns
        -------
        mask : np.ndarray (H, W), dtype bool
        score : float in [0, 1]
        """
        if not self._image_set:
            raise RuntimeError(
                "set_image() must be called before segment_box(). "
                "Or use segment_detections() which handles this for you."
            )

        bbox_arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if bbox_arr.size != 4:
            raise ValueError(f"bbox must have 4 elements [x1,y1,x2,y2]; got {bbox}")

        try:
            if self.backend == "ultralytics":
                return self._segment_box_ultralytics(bbox_arr)
            return self._segment_box_mobilesam(bbox_arr)
        except Exception as e:  # noqa: BLE001
            print(f"[MobileSAMSegmentor] segment_box failed for {bbox_arr.tolist()}: {e}")
            empty = np.zeros(self._image_shape, dtype=bool)
            return empty, 0.0

    def _segment_box_ultralytics(self, bbox: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run a single bbox prompt through the ultralytics SAMPredictor."""
        # ultralytics expects a list-of-lists of [x1, y1, x2, y2].
        results = self.predictor(bboxes=[bbox.tolist()])
        result = results[0]

        if result.masks is None or len(result.masks.data) == 0:
            return np.zeros(self._image_shape, dtype=bool), 0.0

        # masks.data: torch.Tensor of shape (N, H, W), values in {0, 1}
        mask_tensor = result.masks.data[0]
        mask = mask_tensor.cpu().numpy().astype(bool)

        # Some ultralytics versions return masks at the model's working
        # resolution rather than the original image resolution; if so, resize.
        if mask.shape != self._image_shape:
            try:
                import cv2
                mask_uint8 = mask.astype(np.uint8) * 255
                resized = cv2.resize(
                    mask_uint8,
                    (self._image_shape[1], self._image_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                mask = resized > 127
            except Exception:
                pass

        # Mask quality score, when available.
        score = 1.0
        if (
            result.boxes is not None
            and hasattr(result.boxes, "conf")
            and len(result.boxes) > 0
        ):
            try:
                score = float(result.boxes.conf[0].cpu().numpy())
            except Exception:
                score = 1.0

        return mask, score

    def _segment_box_mobilesam(self, bbox: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run a single bbox prompt through the native mobile_sam predictor."""
        masks, scores, _ = self.predictor.predict(
            box=bbox[None, :],          # shape (1, 4)
            multimask_output=False,
        )
        if masks.shape[0] == 0:
            return np.zeros(self._image_shape, dtype=bool), 0.0

        mask = masks[0].astype(bool)
        score = float(scores[0])
        return mask, score

    # ------------------------------------------------------------------
    # Batch over all detections in an image
    # ------------------------------------------------------------------
    def segment_detections(
        self,
        image_rgb: np.ndarray,
        detections: List[Dict],
    ) -> List[Dict]:
        """
        Add a precise mask to every detection in the list.

        Parameters
        ----------
        image_rgb : np.ndarray (H, W, 3) uint8 RGB
        detections : list of dict
            Each dict must contain the key ``'bbox'`` with [x1, y1, x2, y2].
            All other keys are passed through unchanged.

        Returns
        -------
        list of dict
            Same dicts, each with two new keys: ``'mask'`` and ``'mask_score'``.
        """
        if not detections:
            return []

        # Encode the image once — this is the whole point of set_image.
        self.set_image(image_rgb)

        for det in detections:
            if "bbox" not in det:
                raise KeyError(
                    "Each detection must have a 'bbox' key (from YOLODetector.detect)."
                )
            mask, score = self.segment_box(det["bbox"])
            det["mask"] = mask
            det["mask_score"] = score

        return detections

    # ------------------------------------------------------------------
    # Optional: point-prompt API for advanced use
    # ------------------------------------------------------------------
    def segment_point(
        self,
        point: np.ndarray,
        point_label: int = 1,
    ) -> Tuple[np.ndarray, float]:
        """
        Generate a mask from a single point prompt (1=foreground, 0=background).

        Useful for the "Point-To-Tell"–style interaction mentioned in the
        proposal.
        """
        if not self._image_set:
            raise RuntimeError("set_image() must be called before segment_point().")

        pt = np.asarray(point, dtype=np.float32).reshape(-1)
        if pt.size != 2:
            raise ValueError(f"point must have 2 elements [x, y]; got {point}")

        try:
            if self.backend == "ultralytics":
                results = self.predictor(
                    points=[pt.tolist()],
                    labels=[int(point_label)],
                )
                result = results[0]
                if result.masks is None or len(result.masks.data) == 0:
                    return np.zeros(self._image_shape, dtype=bool), 0.0
                mask = result.masks.data[0].cpu().numpy().astype(bool)
                if mask.shape != self._image_shape:
                    import cv2
                    mask_u8 = mask.astype(np.uint8) * 255
                    mask = cv2.resize(
                        mask_u8,
                        (self._image_shape[1], self._image_shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ) > 127
                return mask, 1.0
            else:
                masks, scores, _ = self.predictor.predict(
                    point_coords=pt[None, :],
                    point_labels=np.array([int(point_label)]),
                    multimask_output=False,
                )
                if masks.shape[0] == 0:
                    return np.zeros(self._image_shape, dtype=bool), 0.0
                return masks[0].astype(bool), float(scores[0])
        except Exception as e:  # noqa: BLE001
            print(f"[MobileSAMSegmentor] segment_point failed: {e}")
            return np.zeros(self._image_shape, dtype=bool), 0.0


# ============================================================
# Helper functions
# ============================================================

def mask_to_bbox(mask: np.ndarray) -> np.ndarray:
    """
    Tight axis-aligned bounding box of a binary mask.

    Returns
    -------
    np.ndarray (4,) of dtype uint32 in [x_min, y_min, x_max, y_max] order.
    For an empty mask, returns [0, 0, 0, 0].
    """
    if mask is None or not np.any(mask):
        return np.zeros(4, dtype=np.uint32)

    ys, xs = np.where(mask)
    return np.array(
        [xs.min(), ys.min(), xs.max(), ys.max()],
        dtype=np.uint32,
    )


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Intersection-over-Union between two binary masks of equal shape.
    Returns 0.0 if the union is empty.
    """
    if mask1.shape != mask2.shape:
        raise ValueError(
            f"Mask shape mismatch: {mask1.shape} vs {mask2.shape}"
        )

    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)

    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()

    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def get_mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """
    Centroid (center of mass) of a binary mask, returned as (x, y) floats.
    For an empty mask, returns (0.0, 0.0).
    """
    if mask is None or not np.any(mask):
        return 0.0, 0.0

    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())
