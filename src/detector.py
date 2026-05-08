"""YOLO object detector — thin wrapper over Ultralytics YOLO."""

from typing import Dict, List, Union

import numpy as np

import config


class YOLODetector:
    def __init__(self, model_name=None, confidence=None, iou=None, device=None):
        from ultralytics import YOLO  # heavy import, lazy

        self.model_name = model_name or config.YOLO_MODEL_NAME
        self.confidence = confidence or config.YOLO_CONFIDENCE
        self.iou = iou or config.YOLO_IOU
        self.device = device or config.DEVICE
        self.model = YOLO(self.model_name).to(self.device)
        print(f"YOLODetector: {self.model_name} on {self.device}")

    def detect(self, image: Union[np.ndarray, str]) -> List[Dict]:
        """Return [{bbox, confidence, class_id, class_name}, ...]. bbox is xyxy in pixels."""
        result = self.model(image, conf=self.confidence, iou=self.iou,
                            device=self.device, verbose=False)[0]
        out = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cid = int(box.cls[0].cpu().numpy())
            out.append({
                "bbox": np.array([x1, y1, x2, y2], dtype=np.float32),
                "confidence": float(box.conf[0].cpu().numpy()),
                "class_id": cid,
                "class_name": self.model.names.get(cid, f"class_{cid}"),
            })
        return out
