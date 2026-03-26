"""
Debug script to visualize coordinate matching between YOLO detections and GT boxes
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.detector import YOLODetector
from src.data_loader import DataLoader
from src.evaluation import compute_box_iou

def load_yolo_annotations(label_dir: str) -> dict:
    """Load YOLO format annotations."""
    annotations = {}
    for label_file in Path(label_dir).glob('*.txt'):
        image_name = label_file.stem
        boxes = []
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        boxes.append((x_center, y_center, width, height))
        except:
            continue
        if boxes:
            annotations[image_name] = boxes
    return annotations

def denormalize_bbox(bbox_normalized: tuple, image_shape: tuple) -> np.ndarray:
    """Convert normalized YOLO bbox to pixel coords."""
    x_center, y_center, width, height = bbox_normalized
    h, w = image_shape[:2]
    
    x_center *= w
    y_center *= h
    width *= w
    height *= h
    
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    return np.array([x1, y1, x2, y2])

# Load first training image
data_loader = DataLoader(os.path.join(config.DATA_DIR, 'coco8', 'images', 'train'))
annotations = load_yolo_annotations(os.path.join(config.DATA_DIR, 'coco8', 'labels', 'train'))

detector = YOLODetector(device=config.DEVICE)
sample = data_loader[0]
image = sample['image']
filename = sample['filename']
image_stem = Path(filename).stem

print(f"\n{'='*70}")
print(f"Analyzing: {filename}")
print(f"Image shape: {image.shape}")
print(f"{'='*70}")

# Get YOLO detections
yolo_detections = detector.detect(image)
print(f"\nYOLO Detections: {len(yolo_detections)}")
for idx, det in enumerate(yolo_detections):
    bbox = det['bbox']
    print(f"  [{idx+1}] {det['class_name']:15s} | Conf: {det['confidence']:.3f} | BBox: [{bbox[0]:7.1f}, {bbox[1]:7.1f}, {bbox[2]:7.1f}, {bbox[3]:7.1f}]")

# Get ground truth boxes
if image_stem in annotations:
    gt_boxes_normalized = annotations[image_stem]
    print(f"\nGround Truth Boxes (normalized): {len(gt_boxes_normalized)}")
    
    for idx, gt_norm in enumerate(gt_boxes_normalized):
        gt_pixel = denormalize_bbox(gt_norm, image.shape)
        x_c, y_c, w, h = gt_norm
        print(f"  [{idx+1}] Normalized: [{x_c:.4f}, {y_c:.4f}, {w:.4f}, {h:.4f}] -> Pixel: [{gt_pixel[0]:7.1f}, {gt_pixel[1]:7.1f}, {gt_pixel[2]:7.1f}, {gt_pixel[3]:7.1f}]")
    
    # Try matching
    print(f"\n{'='*70}")
    print("IoU Matching Analysis (IoU >= 0.5)")
    print(f"{'='*70}")
    
    gt_boxes_pixel = [denormalize_bbox(gt, image.shape) for gt in gt_boxes_normalized]
    
    for det_idx, detection in enumerate(yolo_detections):
        pred_box = np.array(detection['bbox'])
        print(f"\nPrediction [{det_idx+1}]: {detection['class_name']}")
        print(f"  Pred box type: {type(pred_box)}, dtype: {pred_box.dtype}")
        print(f"  Pred box: [{pred_box[0]:.1f}, {pred_box[1]:.1f}, {pred_box[2]:.1f}, {pred_box[3]:.1f}]")
        
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes_pixel):
            try:
                iou = compute_box_iou(pred_box, gt_box)
                if iou is None:
                    iou = 0.0
                print(f"    vs GT[{gt_idx+1}]: IoU = {iou:.4f}")
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            except Exception as e:
                print(f"    vs GT[{gt_idx+1}]: ERROR - {e}")
                continue
        
        if best_iou >= 0.5:
            print(f"  ✅ MATCHED to GT[{best_gt_idx+1}] with IoU = {best_iou:.4f}")
        else:
            print(f"  ❌ NO MATCH (best IoU = {best_iou:.4f})")
