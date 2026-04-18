"""
End-to-end depth estimation evaluation on COCO8 dataset.

Loads all COCO8 train/val images, runs depth inference, and generates
depth maps with statistics and visualizations.

Run with: python scripts/evaluate_depth_coco8.py [--dry-run]
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_loader import DataLoader
from src.depth_estimator import DepthEstimator
from src.evaluation import evaluate_depth_qualitative
from src.visualizer import Visualizer


def create_colormap_image(depth_map: np.ndarray, cmap_name: str = 'inferno') -> np.ndarray:
    """
    Convert depth map to RGB image using colormap.

    Args:
        depth_map: (H, W) float array
        cmap_name: Matplotlib colormap name

    Returns:
        (H, W, 3) uint8 RGB image
    """
    # Normalize to [0, 1]
    valid = np.isfinite(depth_map)
    if np.sum(valid) == 0:
        return np.zeros((*depth_map.shape, 3), dtype=np.uint8)

    v_min = np.min(depth_map[valid])
    v_max = np.max(depth_map[valid])

    if v_max <= v_min:
        normalized = np.zeros_like(depth_map)
    else:
        normalized = (depth_map - v_min) / (v_max - v_min)
        normalized = np.clip(normalized, 0, 1)

    # Apply colormap
    cmap = cm.get_cmap(cmap_name)
    colored = cmap(normalized)  # (H, W, 4) RGBA [0, 1]

    # Convert to RGB uint8
    rgb = (colored[..., :3] * 255).astype(np.uint8)
    return rgb


def create_depth_visualization(
    image: np.ndarray,
    depth_map: np.ndarray
) -> np.ndarray:
    """
    Create side-by-side visualization of original image and depth map.

    Args:
        image: (H, W, 3) uint8 RGB original image
        depth_map: (H, W) float relative depth

    Returns:
        (H, 2*W, 3) uint8 RGB side-by-side image
    """
    # Convert depth to colormap
    depth_colored = create_colormap_image(depth_map)

    # Ensure same height
    if image.shape[0] != depth_colored.shape[0]:
        depth_colored = cv2.resize(depth_colored, (image.shape[1], image.shape[0]))

    # Concatenate horizontally
    result = np.hstack([image, depth_colored])
    return result


def process_image(
    estimator: DepthEstimator,
    image_path: str,
    dry_run: bool = False
) -> dict:
    """
    Process single image: estimate depth and compute statistics.

    Args:
        estimator: DepthEstimator instance (or None if dry_run)
        image_path: Path to image file
        dry_run: If True, skip actual inference

    Returns:
        Dict with depth map, stats, and metadata
    """
    # Load image
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"  ERROR loading {image_path}: {e}")
        return None

    # Estimate depth
    start_time = time.time()
    try:
        if not dry_run and estimator is not None:
            depth_map = estimator.estimate_depth(image)
        else:
            depth_map = np.random.rand(*image.shape[:2]).astype(np.float32)
        inference_time = time.time() - start_time
    except Exception as e:
        print(f"  ERROR inferring depth for {image_path}: {e}")
        return None

    # Scale to metric depth (dry run uses default scaling)
    if not dry_run and estimator is not None:
        metric_depth, scale, shift = estimator.scale_depth_to_meters(
            depth_map, gt_depth=None, max_depth=config.MAX_DEPTH_M
        )
    else:
        # Dry-run: simple linear scaling
        metric_depth = depth_map * config.MAX_DEPTH_M
        scale = config.MAX_DEPTH_M
        shift = 0.0

    # Compute statistics
    stats = evaluate_depth_qualitative(metric_depth)
    stats['inference_time_ms'] = inference_time * 1000.0
    stats['scale'] = float(scale)
    stats['shift'] = float(shift)

    return {
        'image': image,
        'depth_map': depth_map,
        'metric_depth': metric_depth,
        'stats': stats,
        'inference_time_ms': stats['inference_time_ms'],
    }


def evaluate_split(
    estimator: DepthEstimator,
    split_name: str,
    image_dir: str,
    output_depth_dir: str,
    dry_run: bool = False
) -> dict:
    """
    Evaluate all images in a dataset split.

    Args:
        estimator: DepthEstimator instance
        split_name: 'train' or 'val'
        image_dir: Directory with images
        output_depth_dir: Directory to save depth visualizations
        dry_run: If True, skip actual inference

    Returns:
        Dict with results for this split
    """
    print(f"\n{'='*70}")
    print(f"Processing {split_name.upper()} Split")
    print(f"{'='*70}")

    # Create output directory
    os.makedirs(output_depth_dir, exist_ok=True)

    # Load images
    try:
        data_loader = DataLoader(image_dir, image_extensions=['.jpg', '.jpeg', '.png'])
    except Exception as e:
        print(f"ERROR loading images from {image_dir}: {e}")
        return None

    if len(data_loader) == 0:
        print(f"No images found in {image_dir}")
        return {
            'split': split_name,
            'num_images': 0,
            'images': {},
        }

    print(f"Found {len(data_loader)} images")

    # Process each image
    results = {
        'split': split_name,
        'num_images': len(data_loader),
        'images': {},
    }

    for idx, sample in enumerate(tqdm(data_loader, desc=f"{split_name}")):
        image_path = sample['path']
        filename = sample['filename']
        stem = Path(filename).stem

        # Process image
        result = process_image(estimator, image_path, dry_run=dry_run)
        if result is None:
            continue

        # Save depth visualization
        viz_image = create_depth_visualization(result['image'], result['depth_map'])
        viz_path = os.path.join(output_depth_dir, f"{stem}_depth.png")
        viz_bgr = cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(viz_path, viz_bgr)

        # Save raw depth map as normalized PNG for reference
        depth_uint8 = (np.clip(result['depth_map'], 0, 1) * 255).astype(np.uint8)
        depth_path = os.path.join(output_depth_dir, f"{stem}_depth_raw.png")
        cv2.imwrite(depth_path, depth_uint8)

        # Store results
        results['images'][stem] = {
            'filename': filename,
            'stats': result['stats'],
            'visualization_path': viz_path,
        }

    # Aggregate statistics
    if results['images']:
        all_stats = [img['stats'] for img in results['images'].values()]
        results['aggregate'] = {
            'num_processed': len(all_stats),
            'avg_mean_depth': float(np.nanmean([s['mean'] for s in all_stats])),
            'avg_std_depth': float(np.nanmean([s['std'] for s in all_stats])),
            'avg_inference_time_ms': float(np.nanmean(
                [s['inference_time_ms'] for s in all_stats]
            )),
        }
    else:
        results['aggregate'] = None

    return results


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description='Evaluate depth estimation on COCO8 dataset'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Skip model loading, only test pipeline structure'
    )
    args = parser.parse_args()

    print("\n" + "="*70)
    print("COCO8 DEPTH ESTIMATION EVALUATION")
    print("="*70)
    print(f"Model: {config.DEPTH_PROCESSOR_NAME}")
    print(f"Device: {config.DEVICE}")
    print(f"Dry-run: {args.dry_run}")

    # Initialize depth estimator
    if not args.dry_run:
        print("\n[1] Loading Depth Estimator...")
        try:
            estimator = DepthEstimator(
                model_name=config.DEPTH_PROCESSOR_NAME,
                device=config.DEVICE
            )
        except ImportError:
            print("ERROR: transformers not installed. Run: pip install transformers")
            return
        except Exception as e:
            print(f"ERROR loading model: {e}")
            return
    else:
        print("\n[1] Skipping model loading (dry-run mode)")
        estimator = None

    # Setup output directories
    depth_maps_base = config.DEPTH_MAPS_DIR
    metrics_dir = config.METRICS_DIR
    os.makedirs(depth_maps_base, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Evaluate train and val splits
    all_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model_name': config.DEPTH_PROCESSOR_NAME,
            'device': config.DEVICE,
            'dry_run': args.dry_run,
        }
    }

    splits = [
        ('train', os.path.join(config.DATA_DIR, 'coco8', 'images', 'train'),
         os.path.join(depth_maps_base, 'coco8_train')),
        ('val', os.path.join(config.DATA_DIR, 'coco8', 'images', 'val'),
         os.path.join(depth_maps_base, 'coco8_val')),
    ]

    total_images = 0
    total_time_ms = 0.0

    for split_name, image_dir, output_dir in splits:
        results = evaluate_split(estimator, split_name, image_dir, output_dir, args.dry_run)
        if results is not None:
            all_results[split_name] = results
            total_images += results['num_images']
            if results['aggregate']:
                total_time_ms += results['aggregate']['avg_inference_time_ms']

    # Save results to JSON
    results_path = os.path.join(metrics_dir, 'coco8_depth_evaluation.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nTotal images processed: {total_images}")
    print(f"Results saved to: {results_path}")
    print(f"Depth maps saved to: {depth_maps_base}")

    # Print summary table
    if 'train' in all_results and all_results['train'].get('aggregate'):
        train_agg = all_results['train']['aggregate']
        print(f"\nTrain Split Summary:")
        print(f"  Images: {all_results['train']['num_images']}")
        print(f"  Avg depth: {train_agg['avg_mean_depth']:.2f} m")
        print(f"  Avg inference: {train_agg['avg_inference_time_ms']:.1f} ms")

    if 'val' in all_results and all_results['val'].get('aggregate'):
        val_agg = all_results['val']['aggregate']
        print(f"\nVal Split Summary:")
        print(f"  Images: {all_results['val']['num_images']}")
        print(f"  Avg depth: {val_agg['avg_mean_depth']:.2f} m")
        print(f"  Avg inference: {val_agg['avg_inference_time_ms']:.1f} ms")

    print(f"\n✓ Evaluation pipeline completed successfully!")
    return all_results


if __name__ == '__main__':
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
