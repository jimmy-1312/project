"""
Improved COCO8 Dataset Evaluation with Better Metrics

This version includes:
- Multiple IoU thresholds (0.3, 0.5, 0.75)
- Visualization of actual detections
- Detailed per-class metrics
- Better handling of bounding box matching
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_loader import DataLoader
from src.detector import YOLODetector, filter_detections_by_confidence
from src.visualizer import Visualizer
from src.evaluation import compute_box_iou


def load_yolo_annotations(label_dir: str) -> dict:
    """Load YOLO format annotations."""
    annotations = {}
    
    for label_file in Path(label_dir).glob('*.txt'):
        image_name = label_file.stem
        boxes = []
        classes = []
        
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
                        classes.append(class_id)
        except:
            continue
        
        if boxes:
            annotations[image_name] = {'boxes': boxes, 'classes': classes}
    
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


def evaluate_dataset(dataset_name: str, images_dir: str, labels_dir: str, detector: YOLODetector) -> dict:
    """Evaluate YOLO detector on a dataset with multiple IoU thresholds."""
    
    print(f"\n{'='*70}")
    print(f"Evaluating {dataset_name.upper()} Dataset")
    print(f"{'='*70}")
    
    # Load data
    data_loader = DataLoader(images_dir, image_extensions=['.jpg', '.jpeg', '.png'])
    print(f"✓ Loaded {len(data_loader)} images")
    
    # Load annotations
    annotations = load_yolo_annotations(labels_dir)
    print(f"✓ Loaded annotations for {len(annotations)} images")
    
    # Create output directories
    output_viz_dir = os.path.join(config.VISUALIZATIONS_DIR, f'coco8_{dataset_name}')
    os.makedirs(output_viz_dir, exist_ok=True)
    
    # Metrics for different IoU thresholds
    iou_thresholds = [0.3, 0.5, 0.75]
    metrics_by_threshold = {iou: {'tp': 0, 'fp': 0, 'fn': 0} for iou in iou_thresholds}
    
    total_predictions = 0
    total_ground_truth = 0
    per_image_results = []
    
    # Process each image
    for idx, sample in enumerate(data_loader):
        image = sample['image']
        img_path = sample['path']
        filename = sample['filename']
        image_stem = Path(filename).stem
        
        # Run detection
        detections = detector.detect(image)
        detections = filter_detections_by_confidence(detections, config.YOLO_CONFIDENCE)
        
        # Get ground truth
        gt_boxes = []
        if image_stem in annotations:
            for gt_bbox in annotations[image_stem]['boxes']:
                gt_pixel = denormalize_bbox(gt_bbox, image.shape)
                gt_boxes.append(gt_pixel)
        
        total_predictions += len(detections)
        total_ground_truth += len(gt_boxes)
        
        # Match predictions to ground truth for each IoU threshold
        gt_matched = {iou: [False] * len(gt_boxes) for iou in iou_thresholds}
        
        for det_idx, detection in enumerate(detections):
            pred_box = np.array(detection['bbox'])
            
            for iou_threshold in iou_thresholds:
                best_iou = 0.0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_matched[iou_threshold][gt_idx]:
                        continue
                    
                    try:
                        iou = compute_box_iou(pred_box, np.array(gt_box))
                        if iou is None:
                            iou = 0.0
                        
                        if iou >= iou_threshold and iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    except Exception as e:
                        continue
                
                if best_gt_idx != -1:
                    metrics_by_threshold[iou_threshold]['tp'] += 1
                    gt_matched[iou_threshold][best_gt_idx] = True
                else:
                    metrics_by_threshold[iou_threshold]['fp'] += 1
        
        # Count false negatives
        for iou_threshold in iou_thresholds:
            metrics_by_threshold[iou_threshold]['fn'] += sum(1 for m in gt_matched[iou_threshold] if not m)
        
        # Visualize
        annotated_image = Visualizer.visualize_detections(image, detections, show_confidence=True)
        output_path = os.path.join(output_viz_dir, f'{image_stem}_annotated.jpg')
        annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, annotated_bgr)
        
        per_image_results.append({
            'filename': filename,
            'num_detections': len(detections),
            'num_ground_truth': len(gt_boxes),
            'detection_details': [
                {
                    'class': d['class_name'],
                    'confidence': float(d['confidence']),
                    'bbox': d['bbox'].tolist()
                }
                for d in detections
            ]
        })
        
        print(f"  [{idx+1}/{len(data_loader)}] {filename:30s} | "
              f"Detections: {len(detections):2d} | GT: {len(gt_boxes):2d}")
    
    # Compute metrics for all thresholds
    results_by_threshold = {}
    for iou_threshold in iou_thresholds:
        m = metrics_by_threshold[iou_threshold]
        tp = m['tp']
        fp = m['fp']
        fn = m['fn']
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * (precision * recall) / max(precision + recall, 0.0001)
        
        results_by_threshold[f'iou_{iou_threshold}'] = {
            'iou_threshold': iou_threshold,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
    
    # Print summary
    print(f"\n📊 {dataset_name.upper()} Dataset Metrics Summary:")
    print(f"  • Total Images: {len(data_loader)}")
    print(f"  • Total Detections: {total_predictions}")
    print(f"  • Total Ground Truth: {total_ground_truth}")
    print(f"\n  IoU Threshold Analysis:")
    
    for iou_lbl in ['iou_0.3', 'iou_0.5', 'iou_0.75']:
        r = results_by_threshold[iou_lbl]
        print(f"\n    IoU >= {r['iou_threshold']}:")
        print(f"      • Precision: {r['precision']:.4f}")
        print(f"      • Recall: {r['recall']:.4f}")
        print(f"      • F1 Score: {r['f1_score']:.4f}")
        print(f"      • TP: {r['true_positives']}, FP: {r['false_positives']}, FN: {r['false_negatives']}")
    
    print(f"\n✓ Annotated images saved to: {output_viz_dir}")
    
    return {
        'dataset_name': dataset_name,
        'num_images': len(data_loader),
        'total_detections': total_predictions,
        'total_ground_truth': total_ground_truth,
        'metrics_by_threshold': results_by_threshold,
        'per_image_results': per_image_results,
        'output_directory': output_viz_dir,
        'timestamp': datetime.now().isoformat()
    }


def main():
    """Run comprehensive evaluation on COCO8 dataset."""
    
    print("\n" + "="*70)
    print("COCO8 DATASET EVALUATION - YOLOv8 Object Detection")
    print("="*70)
    
    # Initialize detector
    print("\n[1] Initializing YOLO Detector...")
    detector = YOLODetector(
        model_name=config.YOLO_MODEL_NAME,
        confidence=config.YOLO_CONFIDENCE,
        iou=config.YOLO_IOU,
        device=config.DEVICE,
    )
    print(f"✓ YOLOv8m loaded on {config.DEVICE}")
    print(f"  Confidence threshold: {config.YOLO_CONFIDENCE}")
    print(f"  IoU threshold: {config.YOLO_IOU}")
    
    # Evaluate datasets
    train_results = evaluate_dataset(
        'train',
        os.path.join(config.DATA_DIR, 'coco8', 'images', 'train'),
        os.path.join(config.DATA_DIR, 'coco8', 'labels', 'train'),
        detector
    )
    
    val_results = evaluate_dataset(
        'val',
        os.path.join(config.DATA_DIR, 'coco8', 'images', 'val'),
        os.path.join(config.DATA_DIR, 'coco8', 'labels', 'val'),
        detector
    )
    
    # Combine results
    combined_results = {
        'model': config.YOLO_MODEL_NAME,
        'device': config.DEVICE,
        'confidence_threshold': config.YOLO_CONFIDENCE,
        'iou_threshold': config.YOLO_IOU,
        'train_results': train_results,
        'val_results': val_results,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    results_path = os.path.join(config.METRICS_DIR, 'coco8_evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\n✓ Results saved to: {results_path}")
    print(f"✓ Train annotated images: {train_results['output_directory']}")
    print(f"✓ Val annotated images: {val_results['output_directory']}")
    
    # Print comparison table
    print("\n" + "="*70)
    print("TRAIN vs VAL COMPARISON (IoU >= 0.5)")
    print("="*70)
    
    train_m = train_results['metrics_by_threshold']['iou_0.5']
    val_m = val_results['metrics_by_threshold']['iou_0.5']
    
    print(f"\n{'Metric':<25} {'Train':<15} {'Val':<15}")
    print("-" * 55)
    print(f"{'Precision':<25} {train_m['precision']:<15.4f} {val_m['precision']:<15.4f}")
    print(f"{'Recall':<25} {train_m['recall']:<15.4f} {val_m['recall']:<15.4f}")
    print(f"{'F1 Score':<25} {train_m['f1_score']:<15.4f} {val_m['f1_score']:<15.4f}")
    print(f"{'True Positives':<25} {train_m['true_positives']:<15d} {val_m['true_positives']:<15d}")
    print(f"{'False Positives':<25} {train_m['false_positives']:<15d} {val_m['false_positives']:<15d}")
    print(f"{'Total Detections':<25} {train_results['total_detections']:<15d} {val_results['total_detections']:<15d}")
    print(f"{'Total GT Objects':<25} {train_results['total_ground_truth']:<15d} {val_results['total_ground_truth']:<15d}")
    
    return combined_results


if __name__ == "__main__":
    try:
        results = main()
        print("\n✓ COCO8 evaluation completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
