"""
Debug: Compare BGR vs RGB image detection
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from src.detector import YOLODetector

image_path = 'data/coco8/images/val/000000000042.jpg'

print("="*80)
print("Testing BGR vs RGB image detection")  
print("="*80)

# Initialize detector
detector = YOLODetector(confidence=0.0)

# Load as BGR (original cv2.imread)
print("\n[1] Load as BGR (cv2.imread):")
img_bgr = cv2.imread(image_path)
print(f"  Shape: {img_bgr.shape}, Dtype: {img_bgr.dtype}")
results_bgr = detector.model(img_bgr, conf=0.0, iou=detector.iou, device=detector.device, verbose=False)
print(f"  Detections: {len(results_bgr[0].boxes) if results_bgr else 0}")
if results_bgr and len(results_bgr[0].boxes) > 0:
    conf = float(results_bgr[0].boxes[0].conf[0].cpu().numpy())
    class_id = int(results_bgr[0].boxes[0].cls[0].cpu().numpy())
    class_name = detector.class_names.get(class_id, f"Class {class_id}")
    print(f"    Top: {class_name} @ {conf:.4f}")

# Load as RGB (cv2.cvtColor BGR2RGB)
print("\n[2] Load as RGB (cv2.cvtColor BGR2RGB):")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
print(f"  Shape: {img_rgb.shape}, Dtype: {img_rgb.dtype}")
results_rgb = detector.model(img_rgb, conf=0.0, iou=detector.iou, device=detector.device, verbose=False)
print(f"  Detections: {len(results_rgb[0].boxes) if results_rgb else 0}")
if results_rgb and len(results_rgb[0].boxes) > 0:
    conf = float(results_rgb[0].boxes[0].conf[0].cpu().numpy())
    class_id = int(results_rgb[0].boxes[0].cls[0].cpu().numpy())
    class_name = detector.class_names.get(class_id, f"Class {class_id}")
    print(f"    Top: {class_name} @ {conf:.4f}")

# Load as BGR but normalized to 0-1
print("\n[3] Load as BGR but normalized to float 0-1:")
img_bgr_norm = img_bgr.astype(np.float32) / 255.0
print(f"  Shape: {img_bgr_norm.shape}, Dtype: {img_bgr_norm.dtype}, Range: [{img_bgr_norm.min():.2f}, {img_bgr_norm.max():.2f}]")
results_norm = detector.model(img_bgr_norm, conf=0.0, iou=detector.iou, device=detector.device, verbose=False)
print(f"  Detections: {len(results_norm[0].boxes) if results_norm else 0}")
if results_norm and len(results_norm[0].boxes) > 0:
    conf = float(results_norm[0].boxes[0].conf[0].cpu().numpy())
    class_id = int(results_norm[0].boxes[0].cls[0].cpu().numpy())
    class_name = detector.class_names.get(class_id, f"Class {class_id}")
    print(f"    Top: {class_name} @ {conf:.4f}")

print("\n" + "="*80)
