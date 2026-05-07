#!/usr/bin/env python3
"""
Duplicate HK indoor train images N times to fight domain dilution when
training on combined HK + public data.

Why:
  Without oversampling, 21 HK images vs 733 Roboflow images = HK is 2.8% of
  the train set. The model fits the public-data distribution and barely
  sees the actual deployment domain (1st-person HK indoor). Naive copy-and-
  paste oversampling is the smallest hammer that works.

What this does:
  For every original HK file in images/train/ (filename starting with one of
  the HK source prefixes), make N additional copies named
  `hk_dup<i>_<original>` (i = 1..N). Same for the matching label file. Same
  for the distances.json entry.

Reproducibility:
  Idempotent. Run with `--clean` to remove all `hk_dup*` files before
  oversampling. Safe to run multiple times.

Usage:
    # 4× oversample → HK ratio rises from ~3% to ~12%
    python3 scripts/oversample_hk.py --copies 4

    # Remove all dup files
    python3 scripts/oversample_hk.py --clean

    # Reset and re-oversample with a different N
    python3 scripts/oversample_hk.py --clean --copies 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, List

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# Filename prefixes that identify our HK-collected images. Anything not
# starting with one of these and not starting with `hk_dup` is treated as
# external (public) data and left untouched.
HK_PREFIXES = ("4471_data__", "comp4471__", "sample__")

DUP_PREFIX = "hk_dup"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def is_hk_original(name: str) -> bool:
    if name.startswith(DUP_PREFIX):
        return False
    return any(name.startswith(p) for p in HK_PREFIXES)


def is_hk_dup(name: str) -> bool:
    return name.startswith(DUP_PREFIX)


def find_hk_originals(img_dir: Path) -> List[Path]:
    return sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in IMG_EXTS and is_hk_original(p.name)
    )


def clean_dups(out_dir: Path) -> int:
    """Remove all files starting with `hk_dup` under images/train and labels/train."""
    n = 0
    for sub in ("images/train", "labels/train"):
        d = out_dir / sub
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if is_hk_dup(f.name):
                f.unlink()
                n += 1
    return n


def update_distances_json(out_dir: Path, new_train: dict, drop_dups_first: bool) -> None:
    p = out_dir / "distances.json"
    if not p.is_file():
        # No distances file → nothing to update; oversampling still works for
        # standard YOLO training (distances are only needed for variant C).
        return
    blob = json.loads(p.read_text())
    train = blob.setdefault("train", {})
    if drop_dups_first:
        train = {k: v for k, v in train.items() if not k.startswith(DUP_PREFIX)}
        blob["train"] = train
    train.update(new_train)
    p.write_text(json.dumps(blob, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out-dir",
        default=os.path.join(config.DATA_DIR, "HK_custom_for_finetuning"),
        help="Dataset root containing images/train and labels/train.",
    )
    parser.add_argument("--copies", type=int, default=4,
                        help="Number of EXTRA copies per HK image (so total = 1 + copies). "
                             "Default 4 → HK density ×5 vs original.")
    parser.add_argument("--clean", action="store_true",
                        help="Remove all hk_dup* files (and their distances.json entries) "
                             "before oversampling. If --copies is 0, just clean.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    img_dir = out_dir / "images" / "train"
    lbl_dir = out_dir / "labels" / "train"
    if not img_dir.is_dir():
        logger.error(f"Image dir not found: {img_dir}")
        sys.exit(1)

    if args.clean:
        n_removed = clean_dups(out_dir)
        update_distances_json(out_dir, {}, drop_dups_first=True)
        logger.info(f"Removed {n_removed} hk_dup* files (and their distances entries).")
        if args.copies <= 0:
            return

    if args.copies <= 0:
        logger.info("--copies <= 0 and --clean not set; nothing to do.")
        return

    # Load existing distances for HK originals so we can replicate them
    distances_blob = {}
    dist_path = out_dir / "distances.json"
    if dist_path.is_file():
        distances_blob = json.loads(dist_path.read_text())
    train_dists_existing = (distances_blob.get("train") or {})

    originals = find_hk_originals(img_dir)
    if not originals:
        logger.error("No HK original images found. Did you run convert_to_yolo.py?")
        sys.exit(1)

    new_dist_entries = {}
    n_img_copies = 0
    n_lbl_copies = 0

    for img in originals:
        stem = img.stem
        ext = img.suffix
        lbl = lbl_dir / f"{stem}.txt"
        existing_dists = train_dists_existing.get(stem)  # may be None

        for i in range(1, args.copies + 1):
            dup_stem = f"{DUP_PREFIX}{i}_{stem}"
            dup_img = img_dir / f"{dup_stem}{ext}"
            dup_lbl = lbl_dir / f"{dup_stem}.txt"

            shutil.copy2(img, dup_img)
            n_img_copies += 1
            if lbl.is_file():
                shutil.copy2(lbl, dup_lbl)
                n_lbl_copies += 1
            if existing_dists is not None:
                new_dist_entries[dup_stem] = list(existing_dists)

    if new_dist_entries:
        update_distances_json(out_dir, new_dist_entries, drop_dups_first=False)

    logger.info("\n" + "=" * 60)
    logger.info(f"HK originals found: {len(originals)}")
    logger.info(f"Copies per original: {args.copies}")
    logger.info(f"Image copies created: {n_img_copies}")
    logger.info(f"Label copies created: {n_lbl_copies}")
    logger.info(f"Distances entries added: {len(new_dist_entries)}")

    # Effective ratio summary
    total = sum(1 for p in img_dir.iterdir()
                if p.suffix.lower() in IMG_EXTS)
    hk_total = sum(1 for p in img_dir.iterdir()
                   if p.suffix.lower() in IMG_EXTS and (is_hk_original(p.name) or is_hk_dup(p.name)))
    logger.info(f"Train images now: {total} (HK: {hk_total}, public: {total - hk_total})")
    logger.info(f"HK ratio: {hk_total / total:.1%}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
