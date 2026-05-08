#!/usr/bin/env python3
"""
Merge an external (Roboflow / Open Images / etc.) YOLO-format dataset into our
HK indoor training set.

Hard rules — see docs/PLAN_PUBLIC_DATA_MERGE.md:

  - Public images go into `images/train` ONLY. Never `val/` or `test/`.
  - Per-target distance for public rows is `null` (JSON), which the dataset
    loader coerces to NaN, which the distance-weighted loss treats as
    "no reweighting" (weight = 1.0).
  - Class names are remapped via `class_mapping.yaml`. Anything mapped to
    `null` is dropped silently. Anything missing from the mapping logs a
    warning and is dropped.
  - Filenames are prefixed `<dataset_name>__<original_stem>` to avoid
    collisions across datasets.

Expected source layout (Roboflow YOLOv8 export):

    <source>/
      data.yaml             # has `names: [class0, class1, ...]`
      train/images/*.{jpg,png,...}
      train/labels/*.txt    # standard 5-col `cls cx cy w h`
      valid/...             # we IGNORE — never goes into our val
      test/...              # we IGNORE

Usage (typical):

    python3 scripts/merge_external_dataset.py \\
        --source ~/Downloads/indoor_furniture_v3 \\
        --dataset-name roboflow_indoor_furniture_v3

    # See what would happen without writing anything:
    python3 scripts/merge_external_dataset.py \\
        --source ~/Downloads/indoor_furniture_v3 \\
        --dataset-name roboflow_indoor_furniture_v3 \\
        --dry-run

    # Re-run cleanly after editing class_mapping.yaml:
    python3 scripts/merge_external_dataset.py \\
        --source ~/Downloads/indoor_furniture_v3 \\
        --dataset-name roboflow_indoor_furniture_v3 \\
        --clean-existing
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ============================================================
# Class taxonomy (must stay in sync with scripts/convert_to_yolo.py)
# ============================================================

from convert_to_yolo import HAZARD_CLASSES, CLASS_TO_ID  # noqa: E402


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ============================================================
# Mapping config
# ============================================================


@dataclass
class DatasetMapping:
    name: str
    source_url: str
    license: str
    notes: str
    classes: Dict[str, Optional[str]]  # external_class_name → our_class_name | None


def load_mapping(mapping_path: str, dataset_name: str) -> DatasetMapping:
    """Load one dataset's mapping block from class_mapping.yaml."""
    import yaml  # lazy

    with open(mapping_path) as f:
        blob = yaml.safe_load(f) or {}
    datasets = blob.get("datasets") or {}
    if dataset_name not in datasets:
        available = sorted(datasets.keys())
        raise KeyError(
            f"dataset {dataset_name!r} not found in {mapping_path}. "
            f"Available: {available or '(none — file is empty/template)'}\n"
            f"Add a new entry to data/HK_custom_for_finetuning/class_mapping.yaml."
        )
    cfg = datasets[dataset_name] or {}

    # Validate that every mapped value is either None or one of OUR classes.
    classes = cfg.get("classes") or {}
    for k, v in classes.items():
        if v is None:
            continue
        if v not in CLASS_TO_ID:
            raise ValueError(
                f"In dataset {dataset_name!r}, class {k!r} maps to {v!r} "
                f"which is not one of our 8 classes ({HAZARD_CLASSES})."
            )

    return DatasetMapping(
        name=dataset_name,
        source_url=str(cfg.get("sourceURL") or ""),
        license=str(cfg.get("license") or ""),
        notes=str(cfg.get("notes") or ""),
        classes=classes,
    )


# ============================================================
# Source dataset parsing
# ============================================================


def load_source_class_names(source_dir: Path) -> List[str]:
    """
    Read `<source>/data.yaml` and return the list of class names by id.

    Roboflow exports use:  names: ['classA', 'classB', ...]   (list)
    Some exports use:       names: {0: 'classA', 1: 'classB', ...}  (dict)
    Both are handled.
    """
    import yaml

    yaml_path = source_dir / "data.yaml"
    if not yaml_path.is_file():
        # Some Roboflow exports put it under `data.yml`
        yaml_path = source_dir / "data.yml"
    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"Neither data.yaml nor data.yml found in {source_dir}"
        )

    blob = yaml.safe_load(yaml_path.read_text()) or {}
    names = blob.get("names")
    if names is None:
        raise ValueError(f"{yaml_path} has no 'names' field")

    if isinstance(names, list):
        return [str(n) for n in names]
    if isinstance(names, dict):
        # Sort by integer key
        return [str(names[k]) for k in sorted(names.keys(), key=int)]
    raise ValueError(f"{yaml_path}: 'names' has unexpected type {type(names).__name__}")


def find_train_pairs(source_dir: Path) -> List[Tuple[Path, Path]]:
    """Return [(image_path, label_path), ...] for the source's train split."""
    img_dir = source_dir / "train" / "images"
    lbl_dir = source_dir / "train" / "labels"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {img_dir} and {lbl_dir} (Roboflow YOLOv8 layout)"
        )

    pairs: List[Tuple[Path, Path]] = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.is_file():
            # Backgrounds (no objects) are valid in YOLO; emit empty-label entry.
            lbl = None
        pairs.append((img, lbl))
    return pairs


# ============================================================
# Label remapping
# ============================================================


@dataclass
class RemappedRow:
    cls_id: int
    cx: float
    cy: float
    w: float
    h: float


def remap_label_file(
    label_path: Optional[Path],
    src_names: List[str],
    mapping: DatasetMapping,
    drop_log: Dict[str, int],
) -> List[RemappedRow]:
    """
    Read 5-column YOLO labels and produce rows with OUR class IDs.

    Drops:
      - Rows whose source class isn't in `mapping.classes`           (key-missing)
      - Rows whose source class is mapped to None                     (null-mapped)
      - Degenerate rows (w<=0 or h<=0)

    Updates `drop_log` (in place): {reason: count}.
    """
    if label_path is None or not label_path.is_file():
        return []

    out: List[RemappedRow] = []
    for raw in label_path.read_text().splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        try:
            src_cls_id = int(float(parts[0]))
            cx, cy, w, h = (float(p) for p in parts[1:5])
        except ValueError:
            drop_log["unparseable_row"] = drop_log.get("unparseable_row", 0) + 1
            continue

        if not (0 <= src_cls_id < len(src_names)):
            drop_log["class_id_out_of_range"] = drop_log.get("class_id_out_of_range", 0) + 1
            continue

        src_name = src_names[src_cls_id]
        if src_name not in mapping.classes:
            drop_log[f"unmapped:{src_name}"] = drop_log.get(f"unmapped:{src_name}", 0) + 1
            continue

        our_name = mapping.classes[src_name]
        if our_name is None:
            drop_log[f"dropped:{src_name}"] = drop_log.get(f"dropped:{src_name}", 0) + 1
            continue

        if w <= 1e-4 or h <= 1e-4:
            drop_log["degenerate_bbox"] = drop_log.get("degenerate_bbox", 0) + 1
            continue

        out.append(
            RemappedRow(
                cls_id=CLASS_TO_ID[our_name],
                cx=cx, cy=cy, w=w, h=h,
            )
        )
    return out


# ============================================================
# Distances.json updater
# ============================================================


def update_distances_json(
    out_dir: Path,
    new_train_entries: Dict[str, List[Optional[float]]],
    keep_existing_dataset_prefix: Optional[str] = None,
) -> None:
    """
    Merge `new_train_entries` into `<out_dir>/distances.json` under the `train` key.

    If `keep_existing_dataset_prefix` is given, all train entries whose stem starts
    with that prefix are first removed (used by --clean-existing).
    """
    p = out_dir / "distances.json"
    if p.is_file():
        blob = json.loads(p.read_text())
    else:
        blob = {"train": {}, "val": {}, "test": {}}

    train = blob.setdefault("train", {})
    if keep_existing_dataset_prefix is not None:
        train = {k: v for k, v in train.items()
                 if not k.startswith(keep_existing_dataset_prefix + "__")}
        blob["train"] = train

    train.update(new_train_entries)
    blob["train"] = train

    p.write_text(json.dumps(blob, indent=2))


def append_license_entry(out_dir: Path, mapping: DatasetMapping, n_images: int) -> None:
    p = out_dir / "licenses.txt"
    line = (
        f"[{mapping.name}] images={n_images}  license={mapping.license!r}  "
        f"url={mapping.source_url}\n"
    )
    with open(p, "a") as f:
        f.write(line)


# ============================================================
# Main
# ============================================================


def clean_existing(out_dir: Path, dataset_name: str) -> int:
    """Remove all train images/labels prefixed with `<dataset>__`. Return count."""
    n = 0
    for sub in ("images/train", "labels/train"):
        d = out_dir / sub
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.name.startswith(f"{dataset_name}__"):
                f.unlink()
                n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True,
                        help="Path to the external Roboflow YOLOv8 dataset folder.")
    parser.add_argument("--dataset-name", required=True,
                        help="Key in class_mapping.yaml (e.g. 'roboflow_indoor_furniture_v3').")
    parser.add_argument("--mapping",
                        default=os.path.join(config.DATA_DIR, "HK_custom_for_finetuning",
                                             "class_mapping.yaml"),
                        help="Path to class_mapping.yaml.")
    parser.add_argument("--out-dir",
                        default=os.path.join(config.DATA_DIR, "HK_custom_for_finetuning"),
                        help="Our HK dataset root (where to inject train images/labels).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen, don't write anything.")
    parser.add_argument("--clean-existing", action="store_true",
                        help="Before merging, remove any prior <dataset>__* files.")
    args = parser.parse_args()

    source_dir = Path(args.source).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    # Load mapping + source schema
    mapping = load_mapping(args.mapping, args.dataset_name)
    src_names = load_source_class_names(source_dir)
    pairs = find_train_pairs(source_dir)

    logger.info(f"Source:        {source_dir}")
    logger.info(f"Dataset name:  {mapping.name}")
    logger.info(f"License:       {mapping.license}")
    logger.info(f"Source URL:    {mapping.source_url}")
    logger.info(f"Source classes (id → name): "
                f"{ {i: n for i, n in enumerate(src_names)} }")
    logger.info(f"Train pairs:   {len(pairs)}")

    # Sanity check: warn about source classes not covered by the mapping
    not_in_mapping = [n for n in src_names if n not in mapping.classes]
    if not_in_mapping:
        logger.warning(
            f"  ⚠ Source classes NOT covered by mapping (will be dropped): "
            f"{not_in_mapping}\n"
            f"     Add them (or set them to null) in class_mapping.yaml to silence."
        )

    if args.clean_existing and not args.dry_run:
        n_removed = clean_existing(out_dir, args.dataset_name)
        logger.info(f"  --clean-existing: removed {n_removed} prior files")

    # Iterate, remap, copy
    drop_log: Dict[str, int] = {}
    new_distances: Dict[str, List[Optional[float]]] = {}
    n_images_kept = 0
    n_rows_kept = 0
    n_images_skipped_no_rows = 0

    if not args.dry_run:
        (out_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in pairs:
        rows = remap_label_file(lbl_path, src_names, mapping, drop_log)
        if not rows:
            n_images_skipped_no_rows += 1
            continue

        prefixed_stem = f"{args.dataset_name}__{img_path.stem}"
        ext = img_path.suffix.lower()
        out_img = out_dir / "images" / "train" / f"{prefixed_stem}{ext}"
        out_lbl = out_dir / "labels" / "train" / f"{prefixed_stem}.txt"

        if not args.dry_run:
            shutil.copy2(img_path, out_img)
            out_lbl.write_text(
                "\n".join(
                    f"{r.cls_id} {r.cx:.6f} {r.cy:.6f} {r.w:.6f} {r.h:.6f}"
                    for r in rows
                ) + "\n"
            )
            new_distances[prefixed_stem] = [None] * len(rows)

        n_images_kept += 1
        n_rows_kept += len(rows)

    if not args.dry_run:
        update_distances_json(
            out_dir,
            new_distances,
            keep_existing_dataset_prefix=args.dataset_name if args.clean_existing else None,
        )
        append_license_entry(out_dir, mapping, n_images_kept)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info(f"Mode:            {'DRY-RUN' if args.dry_run else 'WRITE'}")
    logger.info(f"Images kept:     {n_images_kept}")
    logger.info(f"Labels kept:     {n_rows_kept}")
    logger.info(f"Images skipped (no rows survived mapping): {n_images_skipped_no_rows}")
    if drop_log:
        logger.info("Drop reasons:")
        for k, v in sorted(drop_log.items(), key=lambda kv: -kv[1]):
            logger.info(f"  {v:5d}  {k}")
    if args.dry_run:
        logger.info("\n(no files written — pass --dry-run=false / drop the flag to actually merge)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
