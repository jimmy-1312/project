import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

def compute_box_iou_original(box1: np.ndarray, box2: np.ndarray) -> float:
    """Original function from evaluation.py"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    print(f"  inter_x: [{inter_x_min}, {inter_x_max}]")
    print(f"  inter_y: [{inter_y_min}, {inter_y_max}]")
    print(f"  Check: x_max < x_min? {inter_x_max} < {inter_x_min} = {inter_x_max < inter_x_min}")
    print(f"  Check: y_max < y_min? {inter_y_max} < {inter_y_min} = {inter_y_max < inter_y_min}")
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        print("  -> NO INTERSECTION (early return 0.0)")
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    print(f"  box1_area = ({x1_max} - {x1_min}) * ({y1_max} - {y1_min}) = {box1_area}")
    print(f"  box2_area = ({x2_max} - {x2_min}) * ({y2_max} - {y2_min}) = {box2_area}")
    print(f"  inter_area = {inter_area}")
    print(f"  union_area = {union_area}")
    
    if union_area == 0:
        print("  -> ZERO UNION (return 0.0)")
        return 0.0
    
    iou = inter_area / union_area
    print(f"  IoU = {inter_area} / {union_area} = {iou}")
    return iou

# Test
pred_box = np.array([1.4, 190.4, 620.5, 477.8], dtype=np.float32)
gt_box = np.array([1.1, 187.7, 612.7, 473.5], dtype=np.float32)

print("Testing Prediction [1] vs GT[1]:")
print(f"Pred box: {pred_box}")
print(f"GT box: {gt_box}")
iou = compute_box_iou_original(pred_box, gt_box)
print(f"Result: IoU = {iou}")
