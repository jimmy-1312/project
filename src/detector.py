"""
YOLO-based Object Detection Module

This module provides a wrapper around YOLOv8 for detecting objects in images.

TODO: Implement the YOLODetector class with the following structure:
- __init__: Initialize YOLO model with specified parameters
- detect: Run inference on an image and return detections
- Properties: Access YOLO class names
"""

import numpy as np
from typing import List, Dict, Union, Tuple
import config
from ultralytics import YOLO
import cv2


class YOLODetector:
    """
    YOLO-based object detector using YOLOv8.
    
    This class wraps the Ultralytics YOLO model for convenient object detection.
    It should handle image loading, model inference, and detection formatting.
    
    Attributes:
        model_name (str): Name of YOLO model to load
        confidence (float): Confidence threshold for detections
        iou (float): IOU threshold for NMS (Non-Maximum Suppression)
        device (str): Device to run model on ('cuda' or 'cpu')
        model: The loaded YOLO model object
    """
    
    def __init__(
        self,
        model_name: str = None,
        confidence: float = None,
        iou: float = None,
        device: str = None,
    ) -> None:
        """
        Initialize YOLO detector.
        
        Args:
            model_name (str, optional): YOLO model identifier (e.g., 'yolov8m.pt').
                Defaults to config.YOLO_MODEL_NAME
            confidence (float, optional): Confidence threshold [0-1].
                Defaults to config.YOLO_CONFIDENCE
            iou (float, optional): IOU threshold for NMS [0-1].
                Defaults to config.YOLO_IOU
            device (str, optional): Device name ('cuda'/'cpu').
                Defaults to config.DEVICE
        
        TODO:
        1. Use provided parameters or defaults from config
        2. Load YOLO model using ultralytics
        3. Move model to specified device
        4. Set model to eval mode
        5. Print initialization message
        
        Raises:
            ImportError: If ultralytics/torch not installed
            RuntimeError: If model download/load fails
        """
        self.model_name = model_name or config.YOLO_MODEL_NAME
        self.confidence = confidence or config.YOLO_CONFIDENCE
        self.iou = iou or config.YOLO_IOU
        self.device = device or config.DEVICE
        
        # Load YOLO model
        self.model = YOLO(self.model_name)
        self.model.to(self.device)
        
        print(f"YOLODetector initialized with {self.model_name} on {self.device}")
    
    def detect(self, image: Union[np.ndarray, str]) -> List[Dict]:
        """
        Detect objects in an image.
        
        Args:
            image: Input image - can be:
                - numpy array shape (H, W, 3) in RGB or BGR
                - PIL Image object
                - file path (str) to image
        
        Returns:
            List of detection dictionaries, each containing:
            {
                'bbox': np.array([x1, y1, x2, y2]),  # Top-left and bottom-right corners
                'confidence': float,  # Detection confidence [0-1]
                'class_id': int,  # COCO class ID
                'class_name': str,  # Human-readable class name
            }
        
        TODO:
        1. Run YOLO inference on the image
        2. Extract bounding boxes, scores, and class IDs from results
        3. Filter by confidence threshold
        4. Format output as list of dicts
        5. Return empty list if no detections
        
        Note: YOLO output format varies by version, handle accordingly
        """
        results = self.model(image, conf=self.confidence, iou=self.iou, device=self.device, verbose=False)
        
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names.get(class_id, f"Class {class_id}")
                
                detections.append({
                    'bbox': np.array([x1, y1, x2, y2], dtype=np.float32),
                    'confidence': conf,
                    'class_id': class_id,
                    'class_name': class_name,
                })
        
        return detections
    
    @property
    def class_names(self) -> Dict[int, str]:
        """
        Get mapping of class IDs to class names.
        
        Returns:
            Dict mapping class_id (int) to class_name (str)
        
        TODO:
        1. Access YOLO model's class names attribute
        2. Return as dictionary or convert from list/dict as needed
        """
        return self.model.names
    
    def get_num_classes(self) -> int:
        """
        Get total number of classes the model can detect.
        
        Returns:
            int: Number of classes
        
        TODO:
        1. Return length of class names or model.nc attribute
        """
        return self.model.nc


# ============================================================
# Helper Functions (Optional)
# ============================================================

def convert_xywh_to_xyxy(bbox: np.ndarray) -> np.ndarray:
    """
    Convert bounding box from [x_center, y_center, width, height] to 
    [x1, y1, x2, y2] format.
    
    Args:
        bbox: Array of shape (4,) in [x_c, y_c, w, h] format
    
    Returns:
        Array of shape (4,) in [x1, y1, x2, y2] format
    
    TODO: Implement conversion
    """
    x_c, y_c, w, h = bbox
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return np.array([x1, y1, x2, y2])


def filter_detections_by_class(
    detections: List[Dict],
    class_ids: List[int]
) -> List[Dict]:
    """
    Filter detections to only include specified classes.
    
    Args:
        detections: List of detection dicts from detect()
        class_ids: List of class IDs to keep
    
    Returns:
        Filtered list of detections
    
    TODO: Implement filtering
    """
    return [d for d in detections if d['class_id'] in class_ids]


def filter_detections_by_confidence(
    detections: List[Dict],
    threshold: float
) -> List[Dict]:
    """
    Filter detections by confidence threshold.
    
    Args:
        detections: List of detection dicts from detect()
        threshold: Minimum confidence score [0-1]
    
    Returns:
        Filtered list of detections
    
    TODO: Implement filtering
    """
    return [d for d in detections if d['confidence'] >= threshold]
