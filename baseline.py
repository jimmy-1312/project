"""
Baseline Distance & Direction Evaluator
========================================

Triangle-theorem baseline:
    D = (f * H_real) / H_pixel

Ground-truth format  (info field in every gt_labels JSON)
    info = [H_real_m, gt_clock, gt_d_m]

    H_real_m  -- real height of the object in metres (measured by team)
    gt_clock  -- ground-truth direction as clock hour  9…12…3
    gt_d_m    -- ground-truth distance in metres (measured by team)

Three ready-made gt files (in the same folder as this script):
    gt_labels_sample.json  -- user's own  6 photos  (1.jpg … 6.jpg)
    gt_labels_peter.json   -- Peter's    16 photos  (1.jpg … 16.jpg)
    gt_labels_jimmy.json   -- Jimmy's     6 photos  (image_001.jpg …)

Usage
-----
    # User's photos
    python baseline.py --images-dir . --gt gt_labels_sample.json

    # Peter's photos
    python baseline.py --images-dir . --gt gt_labels_peter.json

    # Jimmy's photos (pixel-bbox mode, 4284x5712)
    python baseline.py --images-dir . --gt gt_labels_jimmy.json --pixel-bbox --img-size 4284 5712
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from Yolo import Yolo26


# ──────────────────────────────────────────────────────────────────────────────
# Camera calibration
# ──────────────────────────────────────────────────────────────────────────────

def focal_length_from_exif(image_path: str) -> Tuple[float, int, int]:
    """
    Auto-compute focal length in pixels from EXIF.
    Returns (f_pixels, img_width, img_height) after EXIF orientation fix.
    """
    img = Image.open(image_path)
    exif = img._getexif() or {}

    f_real_mm = float(exif.get(37386, 5.23))   # FocalLength
    f_35mm    = float(exif.get(41989, 24.0))    # FocalLengthIn35mmFilm

    sensor_width_mm = 36.0 * f_real_mm / f_35mm

    img_corrected  = ImageOps.exif_transpose(img)
    W, H           = img_corrected.size
    pixel_pitch_mm = sensor_width_mm / W
    f_pixels       = f_real_mm / pixel_pitch_mm

    return f_pixels, W, H


# ──────────────────────────────────────────────────────────────────────────────
# Geometry
# ──────────────────────────────────────────────────────────────────────────────

def predict_distance(f_px: float, H_real_cm: float, H_pixel: float) -> float:
    """Returns predicted distance in metres."""
    if H_pixel <= 0:
        return float("nan")
    return (f_px * H_real_cm) / H_pixel / 100.0


def centroid_to_clock(cx: float, img_width: int) -> int:
    """
    Maps centroid x-position to clock direction.
        norm_x=0.0 → 9   (far left)
        norm_x=0.5 → 12  (centre)
        norm_x=1.0 → 3   (far right)
    """
    norm_x    = max(0.0, min(1.0, cx / img_width))
    clock_raw = 12.0 + (norm_x - 0.5) * 6.0   # 9.0 … 15.0
    clock_int = round(clock_raw)
    if clock_int > 12:
        clock_int -= 12
    return clock_int


def clock_error(pred: int, gt: int) -> int:
    """Absolute clock-hour error, handling wrap-around (max 6)."""
    err = abs(pred - gt)
    return min(err, 12 - err)


# ──────────────────────────────────────────────────────────────────────────────
# Per-image evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_image(
    image_path: str,
    gt_objects: List[Dict],
    yolo: Yolo26,
    f_pixels: float,
    img_width: int,
    pixel_bbox: bool = False,
    ref_img_w: int = 1,
    ref_img_h: int = 1,
) -> List[Dict]:
    """
    Run YOLO and match against ground truth objects.

    Parameters
    ----------
    pixel_bbox   : True if gt bboxes are in absolute pixels (Jimmy's dataset)
    ref_img_w/h  : reference image size for pixel_bbox normalisation
    """
    # Skip images with no annotated objects
    if not gt_objects or all(obj.get("class") is None for obj in gt_objects):
        return []

    # Run YOLO
    boxes = yolo.boxes(image_path)
    names = yolo.model.names

    detections = []
    for i, cls_id in enumerate(boxes.cls.tolist()):
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        detections.append({
            "class":   names[int(cls_id)],
            "conf":    float(boxes.conf[i]),
            "H_pixel": y2 - y1,
            "cx":      (x1 + x2) / 2.0,
        })
    detections.sort(key=lambda d: d["conf"], reverse=True)

    results = []
    used = set()

    for gt_obj in gt_objects:
        cls_name = gt_obj.get("class")
        info     = gt_obj.get("info")

        if cls_name is None or info is None:
            continue

        H_real_m, gt_clock, gt_d = info[0], int(info[1]), info[2]
        H_real_cm = H_real_m * 100.0

        # Find best YOLO match for this class
        match_idx = next(
            (i for i, d in enumerate(detections)
             if i not in used and d["class"] == cls_name),
            None,
        )

        if match_idx is None:
            results.append({
                "class":  cls_name,
                "result": [None],
                "_debug": {"gt_d": gt_d, "gt_clock": gt_clock, "H_real_m": H_real_m},
            })
            continue

        used.add(match_idx)
        det = detections[match_idx]

        pred_d     = round(predict_distance(f_pixels, H_real_cm, det["H_pixel"]), 2)
        pred_clock = centroid_to_clock(det["cx"], img_width)
        acc_dist   = round(abs(pred_d - gt_d), 2)
        acc_direc  = clock_error(pred_clock, gt_clock)

        results.append({
            "class":  cls_name,
            "result": [gt_d, pred_d, acc_dist, acc_direc],
            "_debug": {
                "H_pixel":    round(det["H_pixel"], 1),
                "H_real_cm":  H_real_cm,
                "pred_clock": pred_clock,
                "gt_clock":   gt_clock,
                "conf":       round(det["conf"], 3),
            },
        })

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(all_results: Dict, f_pixels: float, save_path: str) -> None:
    lines = [
        "# Baseline Evaluation Report",
        "",
        f"**Method**: Triangle theorem  —  D = (f × H_real) / H_pixel",
        f"**Focal length**: {f_pixels:.1f} px (auto-read from EXIF)",
        "",
        "---",
        "",
        "## Per-Object Results",
        "",
        "| Image | Class | gt_d (m) | pred_d (m) | acc_dist (m) | gt_clk | pred_clk | acc_direc |",
        "|-------|-------|----------|------------|--------------|--------|----------|-----------|",
    ]

    dist_errs, clock_errs = [], []

    for img_name, obj_list in all_results.items():
        for obj in obj_list:
            cls = obj["class"]
            r   = obj["result"]
            dbg = obj.get("_debug", {})

            if r == [None]:
                lines.append(f"| {img_name} | {cls} | {dbg.get('gt_d','—')} | — | — | "
                              f"{dbg.get('gt_clock','—')} | — | — | ❌ |")
                continue

            gt_d, pred_d, acc_dist, acc_direc = r
            lines.append(
                f"| {img_name} | {cls} | {gt_d} | {pred_d} | {acc_dist} | "
                f"{dbg.get('gt_clock','—')} | {dbg.get('pred_clock','—')} | {acc_direc} |"
            )
            if acc_dist  is not None: dist_errs.append(acc_dist)
            if acc_direc is not None: clock_errs.append(acc_direc)

    lines += ["", "---", "", "## Summary Statistics", ""]
    if dist_errs:
        lines.append(f"- **Distance MAE**:  {np.mean(dist_errs):.3f} m")
        lines.append(f"- **Distance RMSE**: {np.sqrt(np.mean(np.square(dist_errs))):.3f} m")
        lines.append(f"- **Distance max**:  {max(dist_errs):.3f} m")
    if clock_errs:
        lines.append(f"- **Direction MAE**: {np.mean(clock_errs):.2f} clock-hours")
        lines.append(f"- **Direction max**: {max(clock_errs)} clock-hours")

    n_total    = sum(len(v) for v in all_results.values())
    n_detected = sum(1 for v in all_results.values()
                     for o in v if o["result"] != [None])
    lines.append(f"- **YOLO detection rate**: {n_detected}/{n_total} "
                 f"({100*n_detected/max(n_total,1):.0f}%)")

    lines += [
        "",
        "---",
        "",
        "## Baseline Limitations (addressed by improved model)",
        "",
        "1. Requires object in YOLO-80 classes — fire, smoke, wet floor undetectable.",
        "2. Single H_real per class — error grows if object is partial or non-standard size.",
        "3. Triangle theorem assumes object is perpendicular — angled objects add error.",
        "4. No depth sensor — Depth Anything addresses all three.",
    ]

    Path(save_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Triangle-theorem baseline evaluator")
    p.add_argument("--images-dir",   default=".",           help="Folder with images")
    p.add_argument("--gt",           default="gt_labels_sample.json",
                                                            help="Ground-truth JSON file")
    p.add_argument("--output",       default="baseline_results.json")
    p.add_argument("--report",       default="baseline_report.md")
    p.add_argument("--scale-factor", type=int, default=1,  help="Yolo26 scale factor")
    p.add_argument("--pixel-bbox",   action="store_true",  help="GT bboxes are in pixels (Jimmy dataset)")
    p.add_argument("--img-size",     type=int, nargs=2, default=[4284, 5712],
                                                           help="Reference WxH for pixel-bbox mode")
    p.add_argument("--focal-px",     type=float, default=None,
                                                           help="Override focal length in pixels")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    images_dir = Path(args.images_dir)
    gt_path    = Path(args.gt)

    if not gt_path.is_file():
        print(f"GT file not found: {gt_path}")
        return 1

    with open(gt_path) as f:
        gt_data = json.load(f)

    # Remove metadata keys
    gt_data = {k: v for k, v in gt_data.items() if not k.startswith("_")}

    print(f"\n{'='*55}")
    print("  YOLO TRIANGLE-THEOREM BASELINE")
    print(f"{'='*55}")
    print(f"  GT file    : {gt_path.name}")
    print(f"  Images dir : {images_dir}")
    print(f"  Images     : {len(gt_data)}")

    # Focal length
    first_img = next(
        (images_dir / n for n in gt_data if (images_dir / n).is_file()), None
    )
    if args.focal_px:
        f_global = args.focal_px
        img_w_global = args.img_size[0]
    elif first_img:
        f_global, img_w_global, _ = focal_length_from_exif(str(first_img))
    else:
        print("No images found in images_dir.")
        return 1

    print(f"  Focal len  : {f_global:.1f} px")

    # Load YOLO
    print(f"\nLoading Yolo26 (scale_factor={args.scale_factor}) ...")
    yolo = Yolo26(scale_factor=args.scale_factor)

    # Evaluate
    all_results: Dict[str, List[Dict]] = {}

    for img_name, gt_objects in gt_data.items():
        img_path = images_dir / img_name
        if not img_path.is_file():
            print(f"  ⚠ Not found: {img_path} — skipping")
            continue

        # Per-image focal length (handles mixed cameras)
        try:
            f_px, img_w, img_h = focal_length_from_exif(str(img_path))
        except Exception:
            f_px, img_w = f_global, img_w_global

        if args.focal_px:
            f_px = args.focal_px

        print(f"\n{img_name}  (f={f_px:.0f}px  w={img_w}px)")

        obj_results = evaluate_image(
            str(img_path), gt_objects, yolo,
            f_px, img_w,
            pixel_bbox=args.pixel_bbox,
            ref_img_w=args.img_size[0],
            ref_img_h=args.img_size[1],
        )

        for obj in obj_results:
            cls = obj["class"]
            r   = obj["result"]
            dbg = obj.get("_debug", {})
            if r == [None]:
                print(f"  ❌ {cls:20s} NOT detected")
            else:
                gt_d, pred_d, acc_dist, acc_direc = r
                print(f"  ✅ {cls:20s}  "
                      f"gt={gt_d}m  pred={pred_d}m  Δd={acc_dist}m  "
                      f"gt_clk={dbg.get('gt_clock')}  pred_clk={dbg.get('pred_clock')}  "
                      f"Δclk={acc_direc}  conf={dbg.get('conf')}")

        all_results[img_name] = obj_results

    # Clean output (no _debug keys)
    clean = {
        img: [{"class": o["class"], "result": o["result"]} for o in objs]
        for img, objs in all_results.items()
    }

    with open(args.output, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\n  Results JSON: {args.output}")

    generate_report(all_results, f_global, args.report)

    # Summary
    dist_errs = [o["result"][2] for objs in all_results.values()
                 for o in objs if o["result"] != [None] and o["result"][2] is not None]
    n_total    = sum(len(v) for v in all_results.values())
    n_detected = sum(1 for v in all_results.values()
                     for o in v if o["result"] != [None])

    print(f"\n{'='*55}")
    if dist_errs:
        print(f"  Distance MAE : {np.mean(dist_errs):.3f} m")
        print(f"  Distance RMSE: {np.sqrt(np.mean(np.square(dist_errs))):.3f} m")
    print(f"  Detected     : {n_detected}/{n_total}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
