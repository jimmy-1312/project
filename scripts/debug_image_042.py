"""
Debug script: Check why dog isn't detected in image 000000000042
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import config
from src.detector import YOLODetector

print("="*80)
print("DEBUG: Why is dog not detected in 000000000042?")
print("="*80)

# Load image
image_path = os.path.join(config.DATA_DIR, 'coco8/images/val/000000000009.jpg')
image = cv2.imread(image_path)

if image is None:
    print(f"❌ Failed to load image: {image_path}")
    sys.exit(1)

print(f"\n✓ Image loaded: {image_path}")
print(f"  Image shape: {image.shape}")

# Initialize detector (same as evaluation script)
print("\n[1] Initialize detector with confidence=0.3 (evaluation config):")
detector = YOLODetector(confidence=config.YOLO_CONFIDENCE)

# Run detection with confidence=0.3 (as evaluation script does)
print("\n[2] Run detection with conf=0.3 (as in evaluation):")
detections = detector.detect(image)
print(f"  Detections found: {len(detections)}")
for i, det in enumerate(detections, 1):
    print(f"    [{i}] {det['class_name']:15s} - Confidence: {det['confidence']:.4f}")

# Now run with raw YOLO (no confidence filtering) to see what's there
print("\n[3] Run YOLO directly with conf=0.0 (raw detections):")
results = detector.model(image, conf=0.0, iou=detector.iou, device=detector.device, verbose=False)

if len(results) > 0:
    result = results[0]
    boxes = result.boxes
    print(f"  Total predictions: {len(boxes)}")
    
    all_preds = []
    for i in range(len(boxes)):
        box = boxes[i]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        class_id = int(box.cls[0].cpu().numpy())
        class_name = detector.class_names.get(class_id, f"Class {class_id}")
        
        all_preds.append({
            'class_name': class_name,
            'class_id': class_id,
            'confidence': conf,
            'bbox': [x1, y1, x2, y2]
        })
    
    # Sort by confidence
    all_preds.sort(key=lambda x: x['confidence'], reverse=True)
    
    for i, pred in enumerate(all_preds[:10], 1):
        status = "✅ PASS 0.3" if pred['confidence'] >= 0.3 else "❌ FAIL 0.3"
        print(f"    [{i}] {pred['class_name']:15s} - {pred['confidence']:.4f} {status}")

print("\n" + "="*80)
print("ANALYSIS:")
print("="*80)

if len(detections) == 0 and len(results) > 0 and len(results[0].boxes) > 0:
    print("\n⚠️  MISMATCH DETECTED!")
    print("  - detector.detect() returned 0 detections")
    print("  - BUT raw YOLO found predictions")
    print("\n📍 Likely cause: Image format issue or DetectionBox parsing bug")
else:
    print("\n✓ Results are consistent")
