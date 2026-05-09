#!/usr/bin/env python3
"""
Stage 1 of HK indoor dataset pipeline: extract images and labels from
the three team-provided docx files into a single normalized JSON.

What this script does:
  - Pulls embedded JPG/PNG bytes from each docx (word/media/*).
  - Parses each docx's `gt_labels_objects` and `gt_labels_environment`
    Python-dict-literal text blocks.
  - Normalizes every bbox to [0, 1] (comp4471 source uses pixel coords).
  - Normalizes every obstacle `info` to [clock, distance_m].
  - Drops `None` annotations.
  - Writes:
      data/HK_custom_for_finetuning/raw_images/<source>__NN.<ext>
      data/HK_custom_for_finetuning/labels_unified.json

Run from project root:
    python3 scripts/build_hk_dataset.py
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ============================================================
# Config — sources and per-source quirks
# ============================================================


@dataclass(frozen=True)
class Source:
    name: str               # short id used in output filenames
    docx_path: str          # absolute path to the docx
    image_count: int        # how many images this docx contributes
    image_key_template: str # how its dict keys are formatted: "{i}.jpg" or "image_{i:03d}.jpg"
    bbox_is_pixel: bool     # True if bboxes are in pixel coordinates
    declared_size: Optional[Tuple[int, int]]  # (W, H) when bboxes are pixel — for normalization
    obstacle_info_format: str  # one of "mixed", "dist_clock", "clock_dist"


def make_sources(uploads_dir: str) -> List[Source]:
    return [
        Source(
            name="sample",
            docx_path=os.path.join(uploads_dir, "sample.docx"),
            image_count=6,
            image_key_template="{i}.jpg",
            bbox_is_pixel=False,
            declared_size=None,
            obstacle_info_format="mixed",
        ),
        Source(
            name="4471_data",
            docx_path=os.path.join(uploads_dir, "4471_data.docx"),
            image_count=16,
            image_key_template="{i}.jpg",
            bbox_is_pixel=False,
            declared_size=None,
            obstacle_info_format="dist_clock",
        ),
        Source(
            name="comp4471",
            docx_path=os.path.join(uploads_dir, "COMP4471 Images.docx"),
            image_count=6,
            image_key_template="image_{i:03d}.jpg",
            bbox_is_pixel=True,
            declared_size=(4284, 5712),
            obstacle_info_format="clock_dist",
        ),
    ]


# ============================================================
# Docx helpers
# ============================================================


def read_docx_text(path: str) -> str:
    """Concatenate all paragraph text from a docx."""
    from docx import Document
    return "\n".join(p.text for p in Document(path).paragraphs)


def extract_dict_block(text: str, var_name: str) -> Optional[str]:
    """
    Find a `var_name = { ... }` assignment in `text` and return the brace-balanced
    block (including the outer braces). Returns None if not found.
    """
    m = re.search(rf"{re.escape(var_name)}\s*=\s*\{{", text)
    if not m:
        return None
    start = m.end() - 1  # at the opening '{'
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_dict_block(block: str) -> Dict[str, Any]:
    """
    Parse a Python dict literal from a docx-extracted block. Strips:
      - `# ...` line comments
      - non-breaking spaces (U+00A0) which docx loves to insert and which
        make ast.literal_eval explode.
    """
    cleaned = re.sub(r"#[^\n]*", "", block).replace("\xa0", " ")
    return ast.literal_eval(cleaned)


def extract_media(zip_path: str, dest_dir: str, source_name: str) -> List[str]:
    """
    Copy embedded media files out of a docx (which is a zip) into `dest_dir`,
    renamed `<source_name>__NN.<ext>`. Returns sorted list of output paths.
    """
    os.makedirs(dest_dir, exist_ok=True)
    out_paths: List[str] = []
    with zipfile.ZipFile(zip_path) as z:
        media = sorted(n for n in z.namelist() if n.startswith("word/media/"))
        for i, name in enumerate(media, 1):
            ext = os.path.splitext(name)[1].lower() or ".jpg"
            out = os.path.join(dest_dir, f"{source_name}__{i:02d}{ext}")
            with z.open(name) as r, open(out, "wb") as w:
                shutil.copyfileobj(r, w)
            out_paths.append(out)
    return out_paths


# ============================================================
# Normalization
# ============================================================


def normalize_bbox(bbox: List[float], img_w: int, img_h: int) -> List[float]:
    """Scale a pixel bbox [x1, y1, x2, y2] to normalized [0, 1]. No-op for already-normalized input."""
    x1, y1, x2, y2 = bbox
    return [float(x1) / img_w, float(y1) / img_h, float(x2) / img_w, float(y2) / img_h]


def normalize_obstacle_info(info: List[float], fmt: str) -> List[float]:
    """
    Map raw obstacle `info` to canonical [clock, distance_m].

    fmt:
      - "clock_dist": already canonical, just clean types
      - "dist_clock": swap to canonical
      - "mixed":     decide per-record by which value is in [1, 12] and integer-like
    """
    a, b = info

    if fmt == "clock_dist":
        return [int(round(a)), float(b)]
    if fmt == "dist_clock":
        return [int(round(b)), float(a)]

    # mixed → heuristic
    a_is_clock = float(a).is_integer() and 1 <= a <= 12
    b_is_clock = float(b).is_integer() and 1 <= b <= 12
    if a_is_clock and not b_is_clock:
        return [int(round(a)), float(b)]
    if b_is_clock and not a_is_clock:
        return [int(round(b)), float(a)]
    if a_is_clock and b_is_clock:
        # Both look like clocks — pick the larger as clock (12 is most common for "ahead").
        clock, dist = (a, b) if a >= b else (b, a)
        return [int(round(clock)), float(dist)]
    # Neither in [1,12] integer → fall back to rounding the first.
    return [int(round(a)), float(b)]


def drop_none(records: List[Dict]) -> List[Dict]:
    """Skip rows where class is None or where bbox is explicitly None (when bbox key exists)."""
    out = []
    for r in records:
        if r.get("class") is None:
            continue
        if "bbox" in r and r.get("bbox") is None:
            continue
        out.append(r)
    return out


# ============================================================
# Per-image ingestion
# ============================================================


def ingest_source(
    source: Source,
    raw_images_dir: str,
) -> Dict[str, Dict[str, Any]]:
    """Extract media, parse dicts, normalize, return {filename: record} for one source."""
    from PIL import Image

    text = read_docx_text(source.docx_path)
    obj_block = extract_dict_block(text, "gt_labels_objects")
    env_block = extract_dict_block(text, "gt_labels_environment")
    if obj_block is None or env_block is None:
        raise ValueError(
            f"{source.name}: failed to find gt_labels_objects/environment in docx"
        )
    objects_dict = parse_dict_block(obj_block)
    environments_dict = parse_dict_block(env_block)

    media_paths = extract_media(source.docx_path, raw_images_dir, source.name)
    if len(media_paths) != source.image_count:
        logger.warning(
            f"{source.name}: expected {source.image_count} images, found {len(media_paths)}"
        )

    out: Dict[str, Dict[str, Any]] = {}

    for i, media_path in enumerate(media_paths, 1):
        key = source.image_key_template.format(i=i)
        fname = os.path.basename(media_path)

        with Image.open(media_path) as im:
            img_w, img_h = im.size

        # Objects
        obj_records = []
        for r in drop_none(objects_dict.get(key, [])):
            bbox = list(r["bbox"])
            if source.bbox_is_pixel:
                W, H = source.declared_size  # type: ignore[misc]
                bbox = normalize_bbox(bbox, W, H)
            obj_records.append(
                {
                    "class": str(r["class"]),
                    "bbox": bbox,
                    "info": [float(v) for v in r["info"]],  # [height, clock, distance]
                }
            )

        # Obstacles
        obs_records = []
        for r in drop_none(environments_dict.get(key, [])):
            bbox = r.get("bbox")
            if bbox is not None and source.bbox_is_pixel:
                W, H = source.declared_size  # type: ignore[misc]
                bbox = normalize_bbox(list(bbox), W, H)
            elif bbox is not None:
                bbox = list(bbox)

            obs_records.append(
                {
                    "class": "obstacle",
                    "bbox": bbox,
                    "info": normalize_obstacle_info(
                        list(r["info"]), source.obstacle_info_format
                    ),
                }
            )

        out[fname] = {
            "source": source.name,
            "img_w": img_w,
            "img_h": img_h,
            "objects": obj_records,
            "obstacles": obs_records,
        }

    return out


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--uploads-dir",
        default=os.path.join(config.DATA_DIR, "source_docx"),
        help="Directory containing the three docx files. Default: data/source_docx/.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(config.DATA_DIR, "HK_custom_for_finetuning"),
        help="Output dataset directory (will create raw_images/ + labels_unified.json).",
    )
    args = parser.parse_args()

    raw_images_dir = os.path.join(args.out_dir, "raw_images")
    out_json = os.path.join(args.out_dir, "labels_unified.json")
    os.makedirs(args.out_dir, exist_ok=True)

    sources = make_sources(args.uploads_dir)
    unified: Dict[str, Dict[str, Any]] = {}
    for s in sources:
        if not os.path.isfile(s.docx_path):
            logger.error(f"{s.name}: docx not found at {s.docx_path}")
            continue
        logger.info(f"Ingesting {s.name} from {os.path.basename(s.docx_path)}")
        unified.update(ingest_source(s, raw_images_dir))

    with open(out_json, "w") as f:
        json.dump(unified, f, indent=2)

    # Summary
    n_obj = sum(len(v["objects"]) for v in unified.values())
    n_obs = sum(len(v["obstacles"]) for v in unified.values())
    n_obs_with_bbox = sum(
        sum(1 for o in v["obstacles"] if o["bbox"] is not None)
        for v in unified.values()
    )
    classes = sorted({o["class"] for v in unified.values() for o in v["objects"]})

    logger.info("\n" + "=" * 60)
    logger.info(f"Wrote {out_json}")
    logger.info(f"Images:    {len(unified)}")
    logger.info(f"Objects:   {n_obj}  (across classes: {classes})")
    logger.info(f"Obstacles: {n_obs}  (with bbox: {n_obs_with_bbox}, without: {n_obs - n_obs_with_bbox})")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
