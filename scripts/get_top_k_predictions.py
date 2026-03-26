"""
Get Top-K Predictions from YOLO for Each Image

This script shows the top 5 (or N) predictions from YOLO for each image,
regardless of confidence threshold. Useful for debugging and analysis.

Usage:
    python scripts/get_top_k_predictions.py --top-k 5 --dataset val
"""

import os
import sys
import argparse
import json
from pathlib import Path
import cv2

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import numpy as np
from src.detector import YOLODetector
from src.data_loader import DataLoader


def get_top_k_predictions(image, detector, top_k=5):
    """
    Get top K predictions from YOLO without confidence filtering.
    
    Args:
        image: Input image (numpy array or path)
        detector: YOLODetector instance
        top_k: Number of top predictions to return
    
    Returns:
        List of dicts with class_name, confidence, bbox
    """
    # Run YOLO with low confidence threshold to catch everything
    results = detector.model(image, conf=0.0, iou=detector.iou, device=detector.device, verbose=False)
    
    predictions = []
    
    if len(results) > 0:
        result = results[0]
        boxes = result.boxes
        
        # Collect all predictions
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = detector.class_names.get(class_id, f"Class {class_id}")
            
            predictions.append({
                'rank': None,  # Will be filled later
                'class_name': class_name,
                'class_id': class_id,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2],
            })
    
    # Sort by confidence (descending)
    predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    # Add rank
    for i, pred in enumerate(predictions[:top_k], 1):
        pred['rank'] = i
    
    # Return only top K
    return predictions[:top_k]


def main():
    parser = argparse.ArgumentParser(description='Get top K predictions for COCO8 images')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top predictions to show')
    parser.add_argument('--dataset', choices=['train', 'val', 'both'], default='val', 
                       help='Which dataset to analyze')
    parser.add_argument('--image', type=str, default=None, 
                       help='Specific image ID to analyze (e.g., 000000000042)')
    
    args = parser.parse_args()
    
    # Initialize detector
    print("[1] Initializing YOLO Detector...")
    detector = YOLODetector(confidence=0.0)  # No filtering
    print(f"✓ YOLOv8m loaded on {detector.device}\n")
    
    # Datasets to process
    datasets = []
    if args.dataset in ['train', 'both']:
        datasets.append(('train', os.path.join(config.DATA_DIR, 'coco8/images/train')))
    if args.dataset in ['val', 'both']:
        datasets.append(('val', os.path.join(config.DATA_DIR, 'coco8/images/val')))
    
    all_results = {}
    
    for dataset_name, dataset_path in datasets:
        print(f"\n{'='*80}")
        print(f"TOP-{args.top_k} PREDICTIONS - {dataset_name.upper()} DATASET")
        print(f"{'='*80}\n")
        
        # Load dataset
        loader = DataLoader(data_dir=dataset_path)
        
        for image_path in Path(dataset_path).glob('*.jpg'):
            image_id = image_path.stem
            
            # Skip if specific image requested and this isn't it
            if args.image and not image_path.stem.endswith(args.image.replace('.png', '').replace('.jpg', '')):
                continue
            
            # Get predictions
            image = cv2.imread(str(image_path))
            top_k_preds = get_top_k_predictions(image, detector, top_k=args.top_k)
            
            # Display results
            print(f"📸 Image: {image_id}.jpg")
            print(f"{'─'*80}")
            
            if top_k_preds:
                for pred in top_k_preds:
                    conf_pct = pred['confidence'] * 100
                    # Color code by confidence
                    if pred['confidence'] >= 0.3:
                        status = "✅ ABOVE 0.3"
                    elif pred['confidence'] >= 0.2:
                        status = "⚠️  0.2-0.3"
                    else:
                        status = "❌ BELOW 0.2"
                    
                    print(f"  [{pred['rank']}] {pred['class_name']:20s} | "
                          f"Confidence: {conf_pct:6.2f}% | {status}")
            else:
                print("  ❌ No predictions")
            
            print()
            
            # Store results
            all_results[image_id] = {
                'dataset': dataset_name,
                'top_k': args.top_k,
                'predictions': [
                    {
                        'rank': p['rank'],
                        'class_name': p['class_name'],
                        'confidence': round(p['confidence'], 4)
                    }
                    for p in top_k_preds
                ]
            }
    
    # Save results to JSON
    output_file = os.path.join(config.RESULTS_DIR, 'metrics', f'top{args.top_k}_predictions.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == '__main__':
    main()
