"""MobileSAM segmentor — refines YOLO bboxes into binary masks via Ultralytics SAM."""

import os
from typing import Dict, List, Tuple

import numpy as np

import config


class MobileSAMSegmentor:
    def __init__(self, checkpoint=None, device=None):
        from ultralytics.models.sam import Predictor as SAMPredictor

        self.checkpoint = checkpoint or config.MOBILE_SAM_CHECKPOINT
        self.device = device or config.DEVICE
        weights = self.checkpoint if os.path.isfile(self.checkpoint) else "mobile_sam.pt"
        self.predictor = SAMPredictor(overrides=dict(
            conf=0.25, task="segment", mode="predict",
            imgsz=1024, model=weights, device=self.device,
            verbose=False, save=False,
        ))
        self._image_shape = None
        print(f"MobileSAMSegmentor: {weights} on {self.device}")

    def set_image(self, image_rgb: np.ndarray) -> None:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"expected (H,W,3) RGB, got {image_rgb.shape}")
        self.predictor.set_image(image_rgb)
        self._image_shape = image_rgb.shape[:2]

    def segment_box(self, bbox) -> Tuple[np.ndarray, float]:
        """[x1,y1,x2,y2] -> (mask HxW bool, score in [0,1])."""
        if self._image_shape is None:
            raise RuntimeError("call set_image() first")
        bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)
        try:
            result = self.predictor(bboxes=[bbox.tolist()])[0]
        except Exception as e:
            print(f"[segmentor] failed for {bbox.tolist()}: {e}")
            return np.zeros(self._image_shape, dtype=bool), 0.0

        if result.masks is None or len(result.masks.data) == 0:
            return np.zeros(self._image_shape, dtype=bool), 0.0

        mask = result.masks.data[0].cpu().numpy().astype(bool)
        if mask.shape != self._image_shape:
            import cv2
            m = cv2.resize(mask.astype(np.uint8) * 255,
                           (self._image_shape[1], self._image_shape[0]),
                           interpolation=cv2.INTER_NEAREST)
            mask = m > 127

        score = 1.0
        try:
            score = float(result.boxes.conf[0].cpu().numpy())
        except Exception:
            pass
        return mask, score

    def segment_detections(self, image_rgb: np.ndarray,
                           detections: List[Dict]) -> List[Dict]:
        """Add 'mask' (bool HxW) and 'mask_score' (float) to each detection. In place."""
        if not detections:
            return detections
        self.set_image(image_rgb)
        for d in detections:
            mask, score = self.segment_box(d["bbox"])
            d["mask"] = mask
            d["mask_score"] = score
        return detections
