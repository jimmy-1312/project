#!/usr/bin/env python3
"""
Restratify train/val/test by re-shuffling ALL files across splits with
source-stratified random sampling.

Why this exists:
  The previous setup put 21 HK photos in train and only 5 in val. The
  resulting val (5 images, 9 instances) gives mAP variance so high that the
  metric oscillates by 0.05–0.10 with single-prediction changes — making
  every "did this experiment help?" question impossible to answer.

  Indoor navigation is the actual goal, not HK-specific performance. So we
  evaluate on a mixed indoor distribution (HK + public Roboflow) and split
  every source the same way (80/10/10). Headline mAP is then statistically
  meaningful, and we can additionally compute HK-only metrics on the HK
  subset of val for the "domain coverage" story.

What this script does:
  1. Collect every image+label currently in images/{train,val,test}/.
  2. Drop any `hk_dup*` files (oversampling should be redone AFTER splitting,
     so the same HK image never appears in both train and val).
  3. Group remaining files by source prefix (HK vs each external dataset).
  4. Per-group, stratified random shuffle with a fixed seed → 80/10/10.
  5. Move image+label files into the new split directories.
  6. Rewrite distances.json by regrouping entries under their new split.
  7. Clear ultralytics' label caches so the next training/eval re-scans.

Output:
  Mutates the dataset in place. Reports a summary of the new split sizes
  and per-source distribution.

Usage:
    python3 scripts/restratify_splits.py
    python3 scripts/restratify_splits.py --val-ratio 0.15 --test-ratio 0.15
    python3 scripts/restratify_splits.py --dry-run    # show what would move
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Filename prefixes that identify HK-collected images (from build_hk_dataset.py).
HK_PREFIXES = ("4471_data__", "comp4471__", "sample__")
DUP_PREFIX = "hk_dup"


def classify_source(name: str) -> str:
    """Return a coarse source label for a filename."""
    if name.startswith(DUP_PREFIX):
        return "HK_DUP"
    for p in HK_PREFIXES:
        if name.startswith(p):
            return "HK"
    # External datasets: prefix `<dataset_name>__`
    if "__" in name:
        return name.split("__", 1)[0]  # e.g. "indoor_furniture_v3"
    return "UNKNOWN"


def collect_all_files(root: Path) -> List[Tuple[Path, Path]]:
    """Return a list of (image_path, label_path) for every file across all splits."""
    pairs: List[Tuple[Path, Path]] = []
    for split in ("train", "val", "test"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            pairs.append((img, lbl))
    return pairs


def stratified_split(
    pairs: List[Tuple[Path, Path]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Tuple[Path, Path]]]:
    """
    Group pairs by source, shuffle each group, and slice into train/val/test
    using the given ratios. Returns {"train": [...], "val": [...], "test": [...]}.
    """
    by_source: Dict[str, List[Tuple[Path, Path]]] = defaultdict(list)
    for img, lbl in pairs:
        src = classify_source(img.name)
        by_source[src].append((img, lbl))

    rng = random.Random(seed)
    train: List[Tuple[Path, Path]] = []
    val: List[Tuple[Path, Path]] = []
    test: List[Tuple[Path, Path]] = []

    for src, items in by_source.items():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * test_ratio)) if n > 0 else 0
        n_val = max(1, round(n * val_ratio)) if n > 0 else 0
        # Guarantee at least 1 train per group
        if n_val + n_test >= n:
            n_val = max(0, n - n_test - 1)

        test_part = items[:n_test]
        val_part = items[n_test : n_test + n_val]
        train_part = items[n_test + n_val :]

        train.extend(train_part)
        val.extend(val_part)
        test.extend(test_part)

    return {"train": train, "val": val, "test": test}


def remove_dup_files(pairs: List[Tuple[Path, Path]]) -> Tuple[List[Tuple[Path, Path]], int]:
    """Filter out hk_dup* files. Returns (kept_pairs, n_removed)."""
    kept: List[Tuple[Path, Path]] = []
    removed_count = 0
    for img, lbl in pairs:
        if img.name.startswith(DUP_PREFIX):
            removed_count += 1
            continue
        kept.append((img, lbl))
    return kept, removed_count


def delete_dup_files_on_disk(root: Path) -> int:
    """Delete the actual hk_dup* files from disk (image + label). Returns count."""
    n = 0
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            d = root / sub / split
            if not d.is_dir():
                continue
            for f in d.iterdir():
                if f.name.startswith(DUP_PREFIX):
                    f.unlink()
                    n += 1
    return n


def remap_distances_json(
    root: Path,
    split_assignment: Dict[str, str],
) -> None:
    """
    Read distances.json, regroup entries under their NEW split, and overwrite.
    `split_assignment` maps each image stem → new split name ("train"|"val"|"test").
    Stems not in the assignment are dropped (e.g. removed dups).
    """
    p = root / "distances.json"
    if not p.is_file():
        return

    blob = json.loads(p.read_text())
    # Flatten: stem → distances list (regardless of which split it WAS in)
    flat: Dict[str, list] = {}
    for split, entries in blob.items():
        if not isinstance(entries, dict):
            continue
        for stem, dists in entries.items():
            flat[stem] = dists

    # Regroup by new assignment
    new = {"train": {}, "val": {}, "test": {}}
    for stem, dists in flat.items():
        new_split = split_assignment.get(stem)
        if new_split is None:
            continue  # dropped (e.g. dup)
        new[new_split][stem] = dists

    p.write_text(json.dumps(new, indent=2))


def clear_label_caches(root: Path) -> int:
    """Delete all `*.cache` files under labels/ to force ultralytics to re-scan."""
    n = 0
    for f in (root / "labels").rglob("*.cache"):
        f.unlink()
        n += 1
    # Sometimes the cache lives at labels/<split>.cache rather than inside the split dir
    for f in (root / "labels").glob("*.cache"):
        if f.is_file():
            f.unlink()
            n += 1
    return n


def write_data_yaml(root: Path, hazard_classes: List[str]) -> Path:
    """(Re)write data.yaml. Schema is identical to convert_to_yolo.py's output."""
    yaml_path = root / "data.yaml"
    yaml_text = (
        "# Auto-generated by scripts/restratify_splits.py — do not edit by hand.\n"
        "# Paths are resolved relative to this file's directory.\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        f"nc: {len(hazard_classes)}\n"
        "names:\n"
        + "\n".join(f"  {i}: {n}" for i, n in enumerate(hazard_classes))
        + "\n"
    )
    yaml_path.write_text(yaml_text)
    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root",
        default=os.path.join(config.DATA_DIR, "HK_custom_for_finetuning"),
        help="Dataset root containing images/{train,val,test}/.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would change without moving any files.",
    )
    args = parser.parse_args()

    # We import HAZARD_CLASSES from convert_to_yolo so taxonomy stays in sync.
    from convert_to_yolo import HAZARD_CLASSES  # noqa: E402

    root = Path(args.root).resolve()
    if not root.is_dir():
        logger.error(f"Dataset root not found: {root}")
        sys.exit(1)

    pairs = collect_all_files(root)
    logger.info(f"Total image+label pairs found: {len(pairs)}")

    # Dups are not split-eligible — drop from logical assignment.
    pairs_no_dup, n_dup = remove_dup_files(pairs)
    if n_dup:
        logger.info(f"  ({n_dup} hk_dup* pairs will be removed; oversample after restratify if desired)")

    # Source breakdown before split
    src_counts = Counter(classify_source(img.name) for img, _ in pairs_no_dup)
    logger.info(f"  Source counts: {dict(src_counts)}")

    splits = stratified_split(
        pairs_no_dup,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Per-source breakdown of the new split
    logger.info("\nNew split sizes:")
    for split_name, items in splits.items():
        per_src = Counter(classify_source(img.name) for img, _ in items)
        logger.info(f"  {split_name:5s}: {len(items):4d}  {dict(per_src)}")

    if args.dry_run:
        logger.info("\n(dry-run — nothing written)")
        return

    # ------------------------------------------------------------------
    # Apply: remove dups, move files into target split dirs, rewrite metadata
    # ------------------------------------------------------------------

    # 1) Delete dup files from disk (their entries in distances.json will also drop)
    n_disk_dup = delete_dup_files_on_disk(root)
    if n_disk_dup:
        logger.info(f"\nRemoved {n_disk_dup} hk_dup* files from disk.")

    # 2) Build assignment: stem → new split
    assignment: Dict[str, str] = {}
    for split_name, items in splits.items():
        for img, _ in items:
            assignment[img.stem] = split_name

    # 3) Move every (image, label) to its new split dir.
    #    We move via a 2-pass strategy with a temp sub-dir to avoid clobbering
    #    when a file's source and dest dirs are the same.
    moves = 0
    for split_name, items in splits.items():
        target_img_dir = root / "images" / split_name
        target_lbl_dir = root / "labels" / split_name
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img, lbl in items:
            new_img = target_img_dir / img.name
            if img != new_img:
                shutil.move(str(img), str(new_img))
                moves += 1
            if lbl is not None and lbl.is_file():
                new_lbl = target_lbl_dir / lbl.name
                if lbl != new_lbl:
                    shutil.move(str(lbl), str(new_lbl))

    logger.info(f"Moved {moves} image files into new split directories.")

    # 4) Regroup distances.json by new assignment
    remap_distances_json(root, assignment)
    logger.info("Updated distances.json.")

    # 5) Refresh data.yaml
    yaml_path = write_data_yaml(root, HAZARD_CLASSES)
    logger.info(f"Wrote {yaml_path}.")

    # 6) Clear ultralytics caches
    n_cache = clear_label_caches(root)
    if n_cache:
        logger.info(f"Cleared {n_cache} ultralytics label cache(s).")

    logger.info("\n" + "=" * 60)
    logger.info("Restratify done. Next steps:")
    logger.info("  (optional) python3 scripts/oversample_hk.py --copies 4")
    logger.info("  python3 scripts/evaluate_hazard.py \\")
    logger.info("      --weights runs/detect/<run>/weights/best.pt \\")
    logger.info("      --conf 0.10 --tag <name> --output results/metrics/<name>.json")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
