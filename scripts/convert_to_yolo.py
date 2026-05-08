#!/usr/bin/env python3
"""
Stage 2 of the HK indoor dataset pipeline: convert the unified labels JSON
(produced by build_hk_dataset.py) into Ultralytics-format YOLO layout.

Per-target distance is stored OUT OF BAND.
  Ultralytics 8.4.x silently rejects detection label rows with more than 5
  columns — training proceeds at "0 instances" with box_loss=0. So we keep
  the standard 5-column format (`cls cx cy w h`) and write distances to a
  parallel JSON file keyed by image stem:

      data/HK_custom_for_finetuning/distances.json
        {
          "train": {"4471_data__01": [1.2, 1.5], ...},
          "val":   {...},
          "test":  {...}
        }

  Distance order matches label-file row order. scripts/evaluate_hazard.py
  reads this file to compute Distance MAE.

Outputs (under args.out_dir):
    images/{train,val,test}/<filename>     copies of the raw images
    labels/{train,val,test}/<stem>.txt     STANDARD 5-column YOLO labels
    distances.json                         out-of-band per-target distances
    splits.json                            which file went into which split
    data.yaml                              Ultralytics dataset config

Run from project root:
    python3 scripts/convert_to_yolo.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ============================================================
# Class taxonomy (final — see docs/PLAN_DEPTH_AWARE_YOLO.md §1)
# ============================================================
# Order matters: the index here becomes the YOLO class id.
HAZARD_CLASSES: List[str] = [
    "chair",          # 0
    "table",          # 1
    "refrigerator",   # 2
    "door",           # 3
    "bed",            # 4
    "couch",          # 5
    "dining_table",   # 6
    "obstacle",       # 7
]
CLASS_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(HAZARD_CLASSES)}

# Map source-string variants to the canonical class names above.
# Anything not in this map is dropped with a warning.
CLASS_ALIASES: Dict[str, str] = {
    "chair": "chair",
    "table": "table",
    "refrigerator": "refrigerator",
    "door": "door",
    "bed": "bed",
    "couch": "couch",
    "sofa": "couch",                   # in case a teammate ever wrote "sofa"
    "dining table": "dining_table",    # space → underscore for the YOLO yaml
    "dining_table": "dining_table",
    "obstacle": "obstacle",
}


# ============================================================
# bbox conversion
# ============================================================


def xyxy_to_yolo(xyxy: List[float]) -> Tuple[float, float, float, float]:
    """Convert [x1, y1, x2, y2] (normalized) to YOLO [cx, cy, w, h] (normalized).
    Caller is responsible for ensuring bbox is already in [0,1]."""
    x1, y1, x2, y2 = xyxy
    # Clip and order (defensive)
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return cx, cy, w, h


# ============================================================
# Per-image label assembly
# ============================================================


def build_label_rows(record: Dict) -> Tuple[List[str], List[float]]:
    """
    Return (label_lines, distances) for one image record.

    label_lines: standard 5-column YOLO format ("cls cx cy w h"). One line
                 per valid target.
    distances:   per-target distance in meters, parallel to label_lines.
                 NaN if not available.

    Skips entries with no bbox or unknown class.
    """
    lines: List[str] = []
    dists: List[float] = []

    # Objects: info = [height, clock, distance]
    for obj in record["objects"]:
        canonical = CLASS_ALIASES.get(obj["class"])
        if canonical is None:
            logger.warning(f"  unknown object class {obj['class']!r}, skipping")
            continue
        cx, cy, w, h = xyxy_to_yolo(obj["bbox"])
        if w <= 1e-4 or h <= 1e-4:
            logger.warning(f"  degenerate bbox for {canonical}, skipping")
            continue
        cls_id = CLASS_TO_ID[canonical]
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        dists.append(float(obj["info"][2]))

    # Obstacles: only those with bbox can be trained on. info = [clock, distance]
    for obs in record["obstacles"]:
        if obs.get("bbox") is None:
            continue
        cx, cy, w, h = xyxy_to_yolo(obs["bbox"])
        if w <= 1e-4 or h <= 1e-4:
            logger.warning(f"  degenerate obstacle bbox, skipping")
            continue
        cls_id = CLASS_TO_ID["obstacle"]
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        dists.append(float(obs["info"][1]))

    return lines, dists


# ============================================================
# Train / val / test split (stratified by source)
# ============================================================


def stratified_split(
    items: List[Tuple[str, Dict]],
    n_test: int,
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Group images by source, then within each group take a slice for test and
    val. Keeps every split mixed across sources.
    """
    rng = random.Random(seed)
    by_source: Dict[str, List[str]] = {}
    for fname, rec in items:
        by_source.setdefault(rec["source"], []).append(fname)

    # Round-robin pick test items across sources so test isn't dominated by one.
    sources = list(by_source.keys())
    pools = {s: rng.sample(by_source[s], len(by_source[s])) for s in sources}
    test = []
    while len(test) < n_test:
        progressed = False
        for s in sources:
            if len(test) >= n_test:
                break
            if pools[s]:
                test.append(pools[s].pop())
                progressed = True
        if not progressed:
            break  # ran out of images everywhere

    # From what remains, val_ratio per source.
    train, val = [], []
    for s, names in pools.items():
        n_val = max(1, round(len(names) * val_ratio)) if names else 0
        val.extend(names[:n_val])
        train.extend(names[n_val:])

    return train, val, test


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--in-json",
        default=os.path.join(config.DATA_DIR, "HK_custom_for_finetuning", "labels_unified.json"),
        help="Path to labels_unified.json from build_hk_dataset.py.",
    )
    parser.add_argument(
        "--raw-images-dir",
        default=os.path.join(config.DATA_DIR, "HK_custom_for_finetuning", "raw_images"),
        help="Directory of extracted raw images (from build_hk_dataset.py).",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(config.DATA_DIR, "HK_custom_for_finetuning"),
        help="Dataset root. Will create images/{train,val,test} and labels/{train,val,test}.",
    )
    parser.add_argument("--n-test", type=int, default=2,
                        help="Number of fully held-out test images. Default 2.")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Validation ratio out of (total - test). Default 0.2.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    args = parser.parse_args()

    with open(args.in_json) as f:
        unified: Dict[str, Dict] = json.load(f)

    if not unified:
        logger.error("labels_unified.json is empty. Did you run build_hk_dataset.py?")
        sys.exit(1)

    items = sorted(unified.items())  # deterministic before shuffle
    train, val, test = stratified_split(
        items, n_test=args.n_test, val_ratio=args.val_ratio, seed=args.seed
    )

    # Write images + labels per split, collect per-target distances out-of-band.
    splits = {"train": train, "val": val, "test": test}
    distances_per_split: Dict[str, Dict[str, List[float]]] = {
        s: {} for s in splits
    }

    for split_name, names in splits.items():
        img_dir = Path(args.out_dir) / "images" / split_name
        lbl_dir = Path(args.out_dir) / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for fname in names:
            rec = unified[fname]
            src_img = Path(args.raw_images_dir) / fname
            if not src_img.is_file():
                logger.error(f"  missing raw image: {src_img}")
                continue
            shutil.copy2(src_img, img_dir / fname)

            stem = Path(fname).stem
            lines, dists = build_label_rows(rec)
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            if dists:
                distances_per_split[split_name][stem] = dists

    # Out-of-band distances for the distance-weighted loss (variant C).
    (Path(args.out_dir) / "distances.json").write_text(
        json.dumps(distances_per_split, indent=2)
    )

    # data.yaml — Ultralytics expects this exact schema.
    # We deliberately omit the `path:` key: when absent, ultralytics resolves
    # train/val/test relative to the yaml file's own directory, which makes
    # the dataset portable across machines (no hard-coded absolute paths).
    yaml_path = Path(args.out_dir) / "data.yaml"
    yaml_text = (
        "# Auto-generated by scripts/convert_to_yolo.py — do not edit by hand.\n"
        "# Paths are resolved relative to this file's directory.\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        f"nc: {len(HAZARD_CLASSES)}\n"
        "names:\n"
        + "\n".join(f"  {i}: {n}" for i, n in enumerate(HAZARD_CLASSES))
        + "\n"
    )
    yaml_path.write_text(yaml_text)

    # Splits manifest
    (Path(args.out_dir) / "splits.json").write_text(
        json.dumps(splits, indent=2)
    )

    logger.info("\n" + "=" * 60)
    logger.info(f"YOLO dataset built at: {args.out_dir}")
    logger.info(f"  classes: {HAZARD_CLASSES}")
    logger.info(f"  train:   {len(train):3d}")
    logger.info(f"  val:     {len(val):3d}")
    logger.info(f"  test:    {len(test):3d}")
    logger.info(f"  total:   {len(train) + len(val) + len(test):3d}")
    logger.info(f"  data.yaml: {yaml_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
