#!/usr/bin/env python3
"""Restructure dataset for k-fold CV on Roboflow + held-out HK test.

After running:
    images/test/              HK only           (held-out, never trained)
    images/roboflow/          Roboflow only     (single pool, k-fold splits dynamically)
    kfold_splits.json         per-fold train/val stem lists
    distances.json            {"test": {hk_stem: [...]}}   HK distances only
    data.yaml                 default points at roboflow/

Idempotent. hk_dup* files (oversample artifacts) are deleted.
"""

import argparse
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
HK_PREFIXES = ("4471_data__", "comp4471__", "sample__")
DUP_PREFIX = "hk_dup"
EXTERNAL_PREFIX = "indoor_furniture_v3__"


def is_hk_original(name: str) -> bool:
    return (not name.startswith(DUP_PREFIX)
            and any(name.startswith(p) for p in HK_PREFIXES))


def is_external(name: str) -> bool:
    return name.startswith(EXTERNAL_PREFIX)


def collect_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """All (image, label) pairs across train/val/test/roboflow/test."""
    pairs = []
    for split in ("train", "val", "test", "roboflow"):
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


def make_kfold(stems: List[str], n_folds: int, seed: int) -> List[Dict]:
    """Deterministic k-fold split → [{k, train: [stems], val: [stems]}, ...]."""
    rng = random.Random(seed)
    shuffled = list(stems)
    rng.shuffle(shuffled)
    n = len(shuffled)
    folds = []
    for k in range(n_folds):
        v_start = (k * n) // n_folds
        v_end = ((k + 1) * n) // n_folds
        val = shuffled[v_start:v_end]
        train = shuffled[:v_start] + shuffled[v_end:]
        folds.append({"k": k, "train": train, "val": val})
    return folds


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root",
                        default=os.path.join(config.DATA_DIR, "HK_custom_for_finetuning"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from convert_to_yolo import HAZARD_CLASSES  # noqa: E402

    root = Path(args.root).resolve()
    pairs = collect_pairs(root)

    hk = [p for p in pairs if is_hk_original(p[0].name)]
    ext = [p for p in pairs if is_external(p[0].name)]
    dup = [p for p in pairs if p[0].name.startswith(DUP_PREFIX)]

    logger.info(f"Found: HK={len(hk)}, Roboflow={len(ext)}, hk_dup={len(dup)} (will delete)")

    if args.dry_run:
        logger.info("(dry-run — nothing moved)")
        return

    # Move HK → test/
    for sub in ("images/test", "labels/test", "images/roboflow", "labels/roboflow"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    for img, lbl in hk:
        dst_img = root / "images" / "test" / img.name
        if img != dst_img:
            shutil.move(str(img), str(dst_img))
        if lbl and lbl.is_file():
            dst_lbl = root / "labels" / "test" / lbl.name
            if lbl != dst_lbl:
                shutil.move(str(lbl), str(dst_lbl))

    # Move Roboflow → roboflow/
    for img, lbl in ext:
        dst_img = root / "images" / "roboflow" / img.name
        if img != dst_img:
            shutil.move(str(img), str(dst_img))
        if lbl and lbl.is_file():
            dst_lbl = root / "labels" / "roboflow" / lbl.name
            if lbl != dst_lbl:
                shutil.move(str(lbl), str(dst_lbl))

    # Delete hk_dup* files
    for img, lbl in dup:
        if img.is_file():
            img.unlink()
        if lbl and lbl.is_file():
            lbl.unlink()

    # Cleanup empty old dirs
    for old in ("train", "val"):
        for sub in ("images", "labels"):
            d = root / sub / old
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()

    # K-fold splits over Roboflow stems
    rb_stems = [p[0].stem for p in ext]
    folds = make_kfold(rb_stems, n_folds=args.n_folds, seed=args.seed)
    (root / "kfold_splits.json").write_text(json.dumps(
        {"n_folds": args.n_folds, "seed": args.seed, "folds": folds}, indent=2,
    ))

    # distances.json: HK only, under "test" key
    flat: Dict[str, list] = {}
    p = root / "distances.json"
    if p.is_file():
        blob = json.loads(p.read_text())
        for entries in blob.values():
            if isinstance(entries, dict):
                flat.update(entries)
    test_dists = {p[0].stem: flat[p[0].stem] for p in hk if p[0].stem in flat}
    p.write_text(json.dumps({"test": test_dists}, indent=2))

    # data.yaml
    yaml_text = (
        "# Auto-generated by scripts/restratify_splits.py.\n"
        "# Default split: roboflow/ as both train and val (overridden by k-fold).\n"
        "train: images/roboflow\n"
        "val: images/roboflow\n"
        "test: images/test\n\n"
        f"nc: {len(HAZARD_CLASSES)}\n"
        "names:\n"
        + "\n".join(f"  {i}: {n}" for i, n in enumerate(HAZARD_CLASSES))
        + "\n"
    )
    (root / "data.yaml").write_text(yaml_text)

    # Clear ultralytics caches
    for c in (root / "labels").rglob("*.cache"):
        c.unlink()

    logger.info("=" * 60)
    logger.info(f"HK held-out test: {len(hk)} → images/test/")
    logger.info(f"Roboflow pool:     {len(ext)} → images/roboflow/")
    logger.info(f"K-fold:            {args.n_folds} folds (~{len(rb_stems)//args.n_folds} val each)")
    logger.info(f"distances.json:    {len(test_dists)} HK entries")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
