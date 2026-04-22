#!/usr/bin/env python
"""
Scene Analysis Demo Script

End-to-end scene analysis on COCO8 dataset or custom images.
Outputs per-image JSON results and PNG visualizations.

Usage:
    # Dry-run (no model downloads)
    python scripts/run_scene_analysis.py --dry-run

    # Dry-run with custom images
    python scripts/run_scene_analysis.py --dry-run --images-dir /path/to/images

    # Real analysis (requires models)
    python scripts/run_scene_analysis.py

    # Real analysis with specific split
    python scripts/run_scene_analysis.py --split train

Run from project root.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.scene_analyzer import analyze_scene


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============================================================
# Stub Implementations for Dry-run
# ============================================================


class StubYOLODetector:
    """
    Dry-run-only stub detector.

    Returns two FIXED, HARDCODED detections ("person" and "car") regardless of
    image contents. Used only with --dry-run to exercise the pipeline without
    downloading/running YOLO. Does NOT reflect what is actually in the image —
    use the real YOLODetector for meaningful output.
    """

    def detect(self, image: np.ndarray) -> List[Dict]:
        H, W = image.shape[:2]
        return [
            {
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.92,
                "bbox": np.array([W * 0.1, H * 0.1, W * 0.4, H * 0.6], dtype=np.float32),
            },
            {
                "class_id": 2,
                "class_name": "car",
                "confidence": 0.88,
                "bbox": np.array([W * 0.5, H * 0.2, W * 0.9, H * 0.7], dtype=np.float32),
            },
        ]


class StubMobileSAMSegmentor:
    """Stub segmentor for dry-run."""

    def segment_detections(
        self, image: np.ndarray, detections: List[Dict]
    ) -> List[Dict]:
        """Add stub masks."""
        H, W = image.shape[:2]
        for i, det in enumerate(detections):
            mask = np.zeros((H, W), dtype=bool)
            x1, y1, x2, y2 = det["bbox"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            mask[y1:y2, x1:x2] = True
            det["mask"] = mask
            det["mask_score"] = 0.90
        return detections


class StubDepthEstimator:
    """Stub depth estimator for dry-run."""

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Return stub depth map."""
        H, W = image.shape[:2]
        # Create a gradient depth map (closer on left, farther on right)
        depth = np.linspace(0.8, 0.2, W)[np.newaxis, :].repeat(H, axis=0)
        return depth.astype(np.float32)

    def scale_depth_to_meters(
        self, depth_map: np.ndarray, **kwargs
    ) -> tuple:
        """Return scaled depth."""
        metric = depth_map * 10.0
        return metric, 10.0, 0.0


# ============================================================
# Visualization
# ============================================================


def visualize_results(
    image: np.ndarray,
    results: List[Dict],
    output_path: str,
) -> None:
    """
    Draw bboxes, masks, and direction indicators on image.

    Args:
        image: Original RGB image
        results: Scene analysis results
        output_path: Where to save annotated image
    """
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for result in results:
        # Draw bbox
        bbox = result["bbox"].astype(int)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw mask overlay
        mask = result["mask"]
        if np.any(mask):
            overlay = img_bgr.copy()
            overlay[mask] = [255, 100, 100]  # Red
            cv2.addWeighted(overlay, 0.3, img_bgr, 0.7, 0, img_bgr)

        # Draw text label.
        # Note: cv2.FONT_HERSHEY_SIMPLEX cannot render the unicode "°" glyph
        # (it gets drawn as "??"), so we use ASCII "deg" instead.
        # For depth, prefer top_k (closest-k mean, robust) > max > mean.
        depth_stats = result.get("depth_stats", {}) or {}
        depth_label = ""
        for key in ("top_k_100", "max", "mean"):
            val = depth_stats.get(key)
            if val is not None and np.isfinite(val):
                depth_label = f" d={val:.2f}"
                break

        text = (
            f"{result['class_name']} {result['confidence']:.2f} "
            f"{result['direction']} {result['angle_deg']:+.1f}deg"
            f"{depth_label}"
        )
        cv2.putText(
            img_bgr,
            text,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        # Draw direction arrow
        centroid_x = int(result["centroid_x_norm"] * image.shape[1])
        centroid_y = (y1 + y2) // 2
        direction = result["direction"]
        if direction == "left":
            cv2.arrowedLine(
                img_bgr, (centroid_x, centroid_y), (centroid_x - 30, centroid_y), (0, 0, 255), 2
            )
        elif direction == "right":
            cv2.arrowedLine(
                img_bgr, (centroid_x, centroid_y), (centroid_x + 30, centroid_y), (0, 0, 255), 2
            )
        elif direction == "center":
            cv2.circle(img_bgr, (centroid_x, centroid_y), 5, (0, 0, 255), -1)

    cv2.imwrite(output_path, img_bgr)
    logger.info(f"Saved visualization: {output_path}")


# ============================================================
# Main
# ============================================================


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_coco8_images() -> List[str]:
    """Find COCO8 images from data directory."""
    coco_dir = os.path.join(config.DATA_DIR, "coco8")
    if not os.path.isdir(coco_dir):
        logger.warning(f"COCO8 directory not found at {coco_dir}")
        return []

    image_paths = []
    for root, _dirs, files in os.walk(coco_dir):
        for f in files:
            if f.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(root, f))

    return sorted(image_paths)


def main():
    parser = argparse.ArgumentParser(description="Scene Analysis Demo")
    parser.add_argument("--dry-run", action="store_true", help="Use stub models (no download)")
    parser.add_argument("--image", type=str, help="Single image file to analyze")
    parser.add_argument("--images-dir", type=str, help="Directory of images to analyze")
    parser.add_argument("--split", choices=["train", "val"], default="train", help="COCO8 split")
    parser.add_argument("--output-dir", type=str, default="results/scene_analysis", help="Output directory")
    parser.add_argument(
        "--max-images", type=int, default=0,
        help="Max images to process (0 = all found). Useful for quick smoke tests."
    )
    args = parser.parse_args()

    # Determine images: --image > --images-dir > COCO8 default
    if args.image:
        if not os.path.isfile(args.image):
            logger.error(f"Image file not found: {args.image}")
            return
        if not args.image.lower().endswith(IMAGE_EXTENSIONS):
            logger.error(f"Unsupported image extension: {args.image}")
            return
        image_paths = [args.image]
    elif args.images_dir:
        if not os.path.isdir(args.images_dir):
            logger.error(f"Image directory not found: {args.images_dir}")
            return
        image_paths = sorted(
            os.path.join(args.images_dir, f)
            for f in os.listdir(args.images_dir)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )
    else:
        image_paths = find_coco8_images()
        if not image_paths:
            logger.error("No images found. Provide --image, --images-dir or ensure COCO8 data is present.")
            return

    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    logger.info(f"Found {len(image_paths)} images")

    # Create output directory
    output_dir = os.path.join(args.output_dir, "images")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize models
    if args.dry_run:
        logger.info("Using stub models (dry-run mode)")
        detector = StubYOLODetector()
        segmentor = StubMobileSAMSegmentor()
        depth_estimator = StubDepthEstimator()
    else:
        logger.info("Loading real models...")
        from src.detector import YOLODetector
        from src.segmentor import MobileSAMSegmentor
        from src.depth_estimator import DepthEstimator
        detector = YOLODetector()
        segmentor = MobileSAMSegmentor()
        depth_estimator = DepthEstimator()

    # Process images
    aggregation_modes = ["mean", "median", "max", {"mode": "top_k", "k": 100}]
    all_results = []

    for i, image_path in enumerate(image_paths, 1):
        logger.info(f"Processing {i}/{len(image_paths)}: {image_path}")

        # Load image: try cv2 first, fall back to PIL (handles WebP, HEIC-as-JPG, etc.)
        image_bgr = cv2.imread(image_path)
        if image_bgr is not None:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            try:
                image_rgb = np.array(Image.open(image_path).convert("RGB"))
            except (UnidentifiedImageError, OSError) as e:
                logger.warning(f"Failed to load (cv2 & PIL): {image_path} ({e})")
                continue

        try:
            # Analyze scene
            results = analyze_scene(
                image_rgb,
                detector,
                segmentor,
                depth_estimator,
                aggregation_modes,
            )

            if results:
                logger.info(f"  Found {len(results)} objects")
                all_results.append({
                    "image_path": image_path,
                    "num_objects": len(results),
                    "results": results,
                })

                # Save JSON
                image_name = Path(image_path).stem
                json_path = os.path.join(output_dir, f"{image_name}_scene.json")
                json_data = {
                    "image_path": image_path,
                    "results": [
                        {
                            "class_id": r["class_id"],
                            "class_name": r["class_name"],
                            "confidence": float(r["confidence"]),
                            "bbox": r["bbox"].tolist(),
                            "mask_score": float(r["mask_score"]),
                            "depth_stats": {k: float(v) for k, v in r["depth_stats"].items()},
                            "direction": r["direction"],
                            "angle_deg": float(r["angle_deg"]),
                            "centroid_x_norm": float(r["centroid_x_norm"]),
                        }
                        for r in results
                    ],
                }
                with open(json_path, "w") as f:
                    json.dump(json_data, f, indent=2)
                logger.info(f"  Saved JSON: {json_path}")

                # Save visualization
                png_path = os.path.join(output_dir, f"{image_name}_overlay.png")
                visualize_results(image_rgb, results, png_path)

            else:
                logger.info("  No objects detected")

        except Exception as e:
            logger.error(f"  Error analyzing {image_path}: {e}")
            import traceback
            traceback.print_exc()

    # Summary report
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Scene Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total images processed: {len(image_paths)}\n")
        f.write(f"Images with detections: {len(all_results)}\n")
        f.write(f"Total objects found: {sum(r['num_objects'] for r in all_results)}\n")
        f.write(f"\nOutput directory: {output_dir}\n")

    logger.info(f"Summary saved to {summary_path}")
    logger.info("Scene analysis complete!")


if __name__ == "__main__":
    main()
