#!/usr/bin/env python3
"""
Precompute metric depth maps for every image in the YOLO dataset.

This makes RGB-D training (variant B) cheap: depth is calculated once
upfront, cached as .npy alongside the image, and loaded by the dataloader
without invoking the depth model in the training loop.

Output layout (matches RGBDYoloDataset's expectations):
    <root>/depth/train/<stem>.npy
    <root>/depth/val/<stem>.npy
    <root>/depth/test/<stem>.npy

Run from project root, AFTER convert_to_yolo.py:
    python3 scripts/precompute_depth.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        default=os.path.join(config.DATA_DIR, "HK_custom_for_finetuning"),
        help="Dataset root containing images/{train,val,test}/.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Splits to process. Default: train val test")
    parser.add_argument("--model", default=None,
                        help="Override config.DEPTH_MODEL_NAME. Default: project config.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if a .npy already exists.")
    args = parser.parse_args()

    # Lazy import — depth model is heavy
    from PIL import Image
    from src.depth_estimator import DepthEstimator

    estimator = DepthEstimator(model_name=args.model)
    if not estimator.is_metric:
        logger.warning(
            "  ⚠ Loaded depth model is RELATIVE, not metric. RGB-D training will "
            "see normalized [0,1] values. If you want true meters, use a "
            "Metric-Indoor variant via --model."
        )

    root = Path(args.root)
    total_processed = 0
    total_skipped = 0

    for split in args.splits:
        img_dir = root / "images" / split
        if not img_dir.is_dir():
            logger.warning(f"  skipping split {split!r}: {img_dir} not found")
            continue
        depth_dir = root / "depth" / split
        depth_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        logger.info(f"\n[{split}] {len(images)} images → {depth_dir}")

        for i, img_path in enumerate(images, 1):
            out = depth_dir / f"{img_path.stem}.npy"
            if out.exists() and not args.overwrite:
                total_skipped += 1
                continue
            try:
                rgb = np.array(Image.open(img_path).convert("RGB"))
            except Exception as e:
                logger.error(f"  [{i}] {img_path.name}: failed to read ({e})")
                continue
            depth = estimator.estimate_depth(rgb).astype(np.float32)
            np.save(out, depth)
            total_processed += 1
            if i % 5 == 0 or i == len(images):
                logger.info(f"  [{i}/{len(images)}] {img_path.name} → {out.name}  "
                            f"shape={depth.shape}, range=[{depth.min():.2f}, {depth.max():.2f}]")

    logger.info("\n" + "=" * 60)
    logger.info(f"Processed: {total_processed}, skipped (already cached): {total_skipped}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
