"""
Debug: Mimic exact evaluation flow for image 000000000042
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import config
from pathlib import Path
from src.detector import YOLODetector, filter_detections_by_confidence
from src.data_loader import DataLoader

print("="*80)
print("DEBUG: Exact evaluation flow for 000000000042")
print("="*80)

# Step 1: Load with DataLoader (as evaluation does)
print("\n[1] Load image with DataLoader (evaluation method):")
val_images_dir = os.path.join(config.DATA_DIR, 'coco8/images/val')
loader = DataLoader(data_dir=val_images_dir)

# Find image
target_image = None
for sample in loader:
    if '000000000042' in sample['filename']:
        target_image = sample
        print(f"  ✓ Found image: {sample['filename']}")
        print(f"    Shape: {sample['image'].shape}")
        print(f"    Dtype: {sample['image'].dtype}")
        break

if target_image is None:
    print("  ❌ Image not found in loader")
    sys.exit(1)

# Step 2: Run detection (as evaluation does)
print("\n[2] Run detection with detector.detect():")
detector = YOLODetector(confidence=config.YOLO_CONFIDENCE)
detections = detector.detect(target_image['image'])
print(f"  Raw detections: {len(detections)}")
for det in detections:
    print(f"    - {det['class_name']:15s} @ {det['confidence']:.4f}")

# Step 3: Apply confidence filter (as evaluation does)
print("\n[3] Apply confidence filter (redundant):")
detections_filtered = filter_detections_by_confidence(detections, config.YOLO_CONFIDENCE)
print(f"  After filtering: {len(detections_filtered)}")
for det in detections_filtered:
    print(f"    - {det['class_name']:15s} @ {det['confidence']:.4f}")

# Step 4: Load ground truth (as evaluation does)
print("\n[4] Load ground truth annotations:")
labels_dir = os.path.join(config.DATA_DIR, 'coco8/labels/val')
label_file = os.path.join(labels_dir, '000000000042.txt')

if os.path.exists(label_file):
    with open(label_file, 'r') as f:
        gt_lines = f.readlines()
    print(f"  ✓ Found {len(gt_lines)} GT objects")
    for line in gt_lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        print(f"    - Class {class_id}")
else:
    print(f"  ❌ Label file not found: {label_file}")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
if len(detections) == 0:
    print("❌ detector.detect() returned 0 detections")
    print("   This suggests the image format from DataLoader is the issue")
elif len(detections_filtered) == 0:
    print("❌ filter_detections_by_confidence removed all detections")
    print("   This is unexpected since dog is at 0.73 > 0.3")
else:
    print(f"✓ Found {len(detections_filtered)} detections")
    print("   The evaluation script SHOULD report this dog!")
