"""
COCO8 Dataset Evaluation Script

Evaluates YOLO detector on real COCO8 training and validation datasets.
Generates annotated images with bounding boxes and comprehensive metrics.

Usage:
    python evaluate_coco8.py
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import yaml

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_loader import DataLoader
from src.detector import YOLODetector, filter_detections_by_confidence
from src.visualizer import Visualizer
from src.evaluation import Evaluator, compute_box_iou


def load_yolo_annotations(label_dir: str) -> dict:
    """
    Load YOLO format annotations (.txt files) for a dataset.
    
    YOLO format: <class_id> <x_center> <y_center> <width> <height> (normalized 0-1)
    """
    annotations = {}
    
    for label_file in Path(label_dir).glob('*.txt'):
        image_name = label_file.stem  # Remove .txt extension
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
        except Exception as e:
            print(f"Warning: Could not load annotations from {label_file}: {e}")
            continue
        
        if boxes:
            annotations[image_name] = {
                'boxes': boxes,
                'classes': classes
            }
    
    return annotations


def denormalize_bbox(bbox_normalized: tuple, image_shape: tuple) -> np.ndarray:
    """
    Convert normalized YOLO bbox (x_center, y_center, width, height) to pixel coords (x1, y1, x2, y2).
    """
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
    """
    Evaluate YOLO detector on a dataset (train or val).
    
    Returns:
        Dictionary with metrics and annotation results
    """
    print(f"\n{'='*70}")
    print(f"Evaluating {dataset_name.upper()} Dataset")
    print(f"{'='*70}")
    
    # Load data
    data_loader = DataLoader(images_dir, image_extensions=['.jpg', '.jpeg', '.png'])
    print(f"\n✓ Loaded {len(data_loader)} images from {dataset_name}")
    
    # Load YOLO annotations
    annotations = load_yolo_annotations(labels_dir)
    print(f"✓ Loaded annotations for {len(annotations)} images")
    
    # Evaluation metrics
    all_predictions = []
    all_gt_detections = []
    per_image_metrics = []
    
    # COCO classes
    coco_classes = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light'
    }
    
    output_viz_dir = os.path.join(config.VISUALIZATIONS_DIR, f'coco8_{dataset_name}')
    os.makedirs(output_viz_dir, exist_ok=True)
    
    # Process each image
    for idx, sample in enumerate(data_loader):
        image = sample['image']
        img_path = sample['path']
        filename = sample['filename']
        image_stem = Path(filename).stem
        
        image_h, image_w = image.shape[:2]
        
        # Run detection
        detections = detector.detect(image)
        detections = filter_detections_by_confidence(detections, config.YOLO_CONFIDENCE)
        
        # Get ground truth
        gt_boxes = []
        gt_classes_list = []
        if image_stem in annotations:
            for gt_bbox, gt_class in zip(annotations[image_stem]['boxes'], 
                                         annotations[image_stem]['classes']):
                gt_pixel = denormalize_bbox(gt_bbox, image.shape)
                gt_boxes.append(gt_pixel)
                gt_classes_list.append(gt_class)
        
        # Prepare predictions for metrics
        pred_boxes = [det['bbox'] for det in detections]
        pred_classes = [det['class_id'] for det in detections]
        pred_scores = [det['confidence'] for det in detections]
        
        # Compute image-level metrics
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes)
        
        for pred_idx, pred_box in enumerate(pred_boxes):
            best_iou = 0.0
            best_gt_idx = -1
            
            # For now, match based on IoU only (not class ID, since YOLO classes might differ from GT)
            for gt_idx, gt_box in enumerate(gt_boxes):
                try:
                    iou = compute_box_iou(np.array(pred_box), np.array(gt_box))
                    if iou is not None and iou > best_iou and iou > 0.5:
                        best_iou = iou
                        best_gt_idx = gt_idx
                except Exception as e:
                    continue
            
            if best_gt_idx != -1:
                tp += 1
                fp -= 1
                fn -= 1
        
        # Image-level precision and recall
        img_precision = tp / max(len(pred_boxes), 1) if len(pred_boxes) > 0 else 0.0
        img_recall = tp / max(len(gt_boxes), 1) if len(gt_boxes) > 0 else 0.0
        
        per_image_metrics.append({
            'filename': filename,
            'num_predictions': len(pred_boxes),
            'num_ground_truth': len(gt_boxes),
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': img_precision,
            'recall': img_recall,
            'avg_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0.0
        })
        
        # Visualize
        annotated_image = Visualizer.visualize_detections(image, detections, show_confidence=True)
        
        output_path = os.path.join(output_viz_dir, f'{image_stem}_annotated.jpg')
        annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, annotated_bgr)
        
        # Print progress
        print(f"  [{idx+1}/{len(data_loader)}] {filename:30s} | "
              f"Detections: {len(detections):2d} | GT: {len(gt_boxes):2d} | "
              f"Precision: {img_precision:.3f} | Recall: {img_recall:.3f}")
        
        # Store for dataset metrics
        all_predictions.append({
            'filename': filename,
            'predictions': [{
                'bbox': d['bbox'].tolist(),
                'class_id': d['class_id'],
                'class_name': d['class_name'],
                'confidence': d['confidence']
            } for d in detections],
            'ground_truth': [{
                'bbox': box.tolist(),
                'class_id': cls_id
            } for box, cls_id in zip(gt_boxes, gt_classes_list)]
        })
    
    # Compute dataset-level metrics
    total_tp = sum(m['true_positives'] for m in per_image_metrics)
    total_fp = sum(m['false_positives'] for m in per_image_metrics)
    total_fn = sum(m['false_negatives'] for m in per_image_metrics)
    total_predictions = sum(m['num_predictions'] for m in per_image_metrics)
    total_ground_truth = sum(m['num_ground_truth'] for m in per_image_metrics)
    
    dataset_precision = total_tp / max(total_predictions, 1)
    dataset_recall = total_tp / max(total_ground_truth, 1)
    dataset_f1 = 2 * (dataset_precision * dataset_recall) / max(dataset_precision + dataset_recall, 0.0001)
    
    results = {
        'dataset_name': dataset_name,
        'num_images': len(data_loader),
        'metrics': {
            'total_predictions': total_predictions,
            'total_ground_truth': total_ground_truth,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn,
            'precision': float(dataset_precision),
            'recall': float(dataset_recall),
            'f1_score': float(dataset_f1),
            'avg_confidence': float(np.mean([m['avg_confidence'] for m in per_image_metrics]))
        },
        'per_image_metrics': per_image_metrics,
        'output_directory': output_viz_dir,
        'timestamp': datetime.now().isoformat()
    }
    
    # Print summary
    print(f"\n📊 {dataset_name.upper()} Dataset Metrics Summary:")
    print(f"  • Total Images: {len(data_loader)}")
    print(f"  • Total Predictions: {total_predictions}")
    print(f"  • Total Ground Truth: {total_ground_truth}")
    print(f"  • True Positives: {total_tp}")
    print(f"  • False Positives: {total_fp}")
    print(f"  • False Negatives: {total_fn}")
    print(f"  • Precision: {dataset_precision:.4f}")
    print(f"  • Recall: {dataset_recall:.4f}")
    print(f"  • F1 Score: {dataset_f1:.4f}")
    print(f"  • Avg Confidence: {np.mean([m['avg_confidence'] for m in per_image_metrics]):.4f}")
    print(f"\n✓ Annotated images saved to: {output_viz_dir}")
    
    return results


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
    
    # Evaluate training dataset
    train_results = evaluate_dataset(
        'train',
        os.path.join(config.DATA_DIR, 'coco8', 'images', 'train'),
        os.path.join(config.DATA_DIR, 'coco8', 'labels', 'train'),
        detector
    )
    
    # Evaluate validation dataset
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
    
    # Print comparison
    print("\n" + "="*70)
    print("TRAIN vs VAL COMPARISON")
    print("="*70)
    print(f"\n{'Metric':<25} {'Train':<15} {'Val':<15}")
    print("-" * 55)
    print(f"{'Precision':<25} {train_results['metrics']['precision']:<15.4f} {val_results['metrics']['precision']:<15.4f}")
    print(f"{'Recall':<25} {train_results['metrics']['recall']:<15.4f} {val_results['metrics']['recall']:<15.4f}")
    print(f"{'F1 Score':<25} {train_results['metrics']['f1_score']:<15.4f} {val_results['metrics']['f1_score']:<15.4f}")
    print(f"{'Avg Confidence':<25} {train_results['metrics']['avg_confidence']:<15.4f} {val_results['metrics']['avg_confidence']:<15.4f}")
    print(f"{'Total Predictions':<25} {train_results['metrics']['total_predictions']:<15d} {val_results['metrics']['total_predictions']:<15d}")
    print(f"{'Total Ground Truth':<25} {train_results['metrics']['total_ground_truth']:<15d} {val_results['metrics']['total_ground_truth']:<15d}")
    
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
