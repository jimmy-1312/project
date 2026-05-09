from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from Yolo import Yolo26


# ──────────────────────────────────────────────────────────────────────────────
# Real-world object heights (cm) — one value per YOLO class name.
# ──────────────────────────────────────────────────────────────────────────────
KNOWN_HEIGHTS: Dict[str, float] = {
    "refrigerator":  180.0,
    "oven":           85.0,
    "microwave":      30.0,
    "tv":             70.0,
    "laptop":         30.0,
    "chair":          90.0,
    "couch":          85.0,
    "bed":            50.0,
    "dining table":   75.0,
    "dining_table":   75.0,
    "toilet":         40.0,
    "sink":           85.0,
    "door":          200.0,
    "person":        170.0,
    "dog":            50.0,
    "cat":            25.0,
    "bottle":         25.0,
    "cup":            10.0,
    "bowl":            8.0,
    "vase":           30.0,
    "suitcase":       65.0,
    "backpack":       50.0,
    "bicycle":       100.0,
    "motorcycle":    110.0,
    "car":           150.0,
    "bus":           300.0,
    "truck":         250.0,
    "table":          75.0,
    "obstacle":      170.0,
}

# Direction thresholds — match config.py exactly
DIR_LEFT  = 0.33
DIR_RIGHT = 0.67


# ──────────────────────────────────────────────────────────────────────────────
# Camera calibration (auto-read from EXIF)
# ──────────────────────────────────────────────────────────────────────────────

def focal_length_from_exif(image_path: str) -> Tuple[float, int, int]:
    """
    Compute focal length in pixels from EXIF.
    Returns (f_pixels, img_width, img_height) after EXIF orientation fix.
    Verified on Samsung SM-A5360: f=2312 px.
    """
    img = Image.open(image_path)
    exif = img._getexif() or {}

    f_real_mm = float(exif.get(37386, 5.23))   # FocalLength tag
    f_35mm    = float(exif.get(41989, 24.0))    # FocalLengthIn35mmFilm tag

    sensor_width_mm = 36.0 * f_real_mm / f_35mm
    img_corrected   = ImageOps.exif_transpose(img)
    W, H            = img_corrected.size
    f_pixels        = f_real_mm / (sensor_width_mm / W)

    return f_pixels, W, H


# ──────────────────────────────────────────────────────────────────────────────
# Geometry — mirrors scene_analyzer.py exactly
# ──────────────────────────────────────────────────────────────────────────────

def predict_distance_m(f_px: float, H_real_cm: float, H_pixel: float) -> float:
    """Triangle theorem.  D (metres) = (f × H_real) / H_pixel."""
    if H_pixel <= 0:
        return float("nan")
    return (f_px * H_real_cm) / H_pixel / 100.0


def centroid_to_direction(cx: float, img_width: int) -> Tuple[str, float]:
    """
    Convert bounding-box centroid x → (direction, angle_deg).

    Mirrors _compute_direction_and_angle() in scene_analyzer.py:
        norm_x < DIR_LEFT  → "left"
        norm_x > DIR_RIGHT → "right"
        else               → "center"
    angle_deg = (norm_x - 0.5) * HORIZONTAL_FOV  (FOV = 60 deg from config.py)
    """
    norm_x    = float(np.clip(cx / max(img_width, 1), 0.0, 1.0))
    angle_deg = (norm_x - 0.5) * 60.0

    if norm_x < DIR_LEFT:
        direction = "left"
    elif norm_x > DIR_RIGHT:
        direction = "right"
    else:
        direction = "center"

    return direction, angle_deg


# ──────────────────────────────────────────────────────────────────────────────
# Alert formatter — mirrors format_nearest_alert() in proximity_alerter.py
# ──────────────────────────────────────────────────────────────────────────────
_DIRECTION_WORD = {"left": "left", "center": "ahead", "right": "right"}


def format_alert(class_name: str, direction: str, distance_m: float) -> str:
    """
    Produces "{direction_word} {dist}m {class}" — identical to proximity_alerter.py.
    Examples: "ahead 1.1m refrigerator",  "left 0.8m chair"
    """
    dir_word = _DIRECTION_WORD.get(direction, "")
    dist_str = f"{distance_m:.1f}m" if math.isfinite(distance_m) and distance_m > 0 else ""
    parts    = [p for p in [dir_word, dist_str, class_name] if p]
    return " ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Clock helpers — ground-truth uses clock hours; convert to direction
# ──────────────────────────────────────────────────────────────────────────────

def clock_to_direction(clock: int) -> str:
    """Map clock hour → direction using same zone boundaries as scene_analyzer."""
    if clock in (9, 10):
        return "left"
    if clock in (2, 3):
        return "right"
    return "center"  # 11, 12, 1


def direction_zone_error(pred: str, gt: str) -> int:
    """0 = exact, 1 = adjacent zone, 2 = opposite ends."""
    order = ["left", "center", "right"]
    try:
        return abs(order.index(pred) - order.index(gt))
    except ValueError:
        return -1


# ──────────────────────────────────────────────────────────────────────────────
# Per-image evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_image(
    image_path: str,
    gt_objects: List[Dict],
    yolo: Yolo26,
    f_pixels: float,
    img_width: int,
) -> List[Dict]:
    """
    Run YOLO on one image and match each GT object to the best detection.

    Returns one dict per GT object shaped to match evaluate.py's per-image
    records — including depth_stats["top_k_100"] so downstream comparison
    code can read distance without knowing which evaluator produced the file.
    """
    if not gt_objects or all(o.get("class") is None for o in gt_objects):
        return []

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

        H_real_m, gt_clock, gt_d = float(info[0]), int(info[1]), float(info[2])
        H_real_cm    = H_real_m * 100.0
        gt_direction = clock_to_direction(gt_clock)

        match_idx = next(
            (i for i, d in enumerate(detections)
             if i not in used and d["class"] == cls_name),
            None,
        )

        if match_idx is None:
            results.append({
                "class":        cls_name,
                "gt_d":         gt_d,
                "gt_direction": gt_direction,
                "pred_d":       None,
                "direction":    None,
                "angle_deg":    None,
                "confidence":   None,
                "depth_stats":  None,   # same key name as scene_analyzer output
                "alert":        None,
                "acc_dist":     None,
                "dir_error":    None,
                "detected":     False,
            })
            continue

        used.add(match_idx)
        det = detections[match_idx]

        pred_d    = round(predict_distance_m(f_pixels, H_real_cm, det["H_pixel"]), 2)
        direction, angle_deg = centroid_to_direction(det["cx"], img_width)
        alert     = format_alert(cls_name, direction, pred_d)
        acc_dist  = round(abs(pred_d - gt_d), 2) if math.isfinite(pred_d) else None
        d_err     = direction_zone_error(direction, gt_direction)

        results.append({
            "class":        cls_name,
            "gt_d":         gt_d,
            "gt_direction": gt_direction,
            "pred_d":       pred_d,
            "direction":    direction,
            "angle_deg":    round(angle_deg, 1),
            "confidence":   round(det["conf"], 3),
            # depth_stats key matches scene_analyzer / evaluate.py exactly
            "depth_stats":  {"top_k_100": pred_d},
            "alert":        alert,
            "acc_dist":     acc_dist,
            "dir_error":    d_err,
            "detected":     True,
        })

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Summary JSON — same schema as evaluate.py for direct comparison
# ──────────────────────────────────────────────────────────────────────────────

def build_summary_json(
    all_results: Dict[str, List[Dict]],
    f_pixels: float,
    gt_file: str,
) -> Dict:
    dist_errs, dir_errs = [], []
    per_image = []

    for img_name, obj_list in all_results.items():
        n_det = sum(1 for o in obj_list if o["detected"])

        for obj in obj_list:
            if obj["acc_dist"] is not None:
                dist_errs.append(obj["acc_dist"])
            if obj["dir_error"] is not None and obj["dir_error"] >= 0:
                dir_errs.append(obj["dir_error"])

        per_image.append({
            "image":      img_name,
            "n_gt":       len(obj_list),
            "n_detected": n_det,
            "objects":    obj_list,
        })

    return {
        # Identification
        "tag":             "baseline_triangle_theorem",
        "method":          "YOLO26 + triangle theorem (D = f·H_real/H_pixel)",
        "gt_file":         gt_file,
        "focal_length_px": round(f_pixels, 1),

        # Core metrics — same keys as evaluate.py
        "distance_MAE_m":      round(float(np.mean(dist_errs)), 4) if dist_errs else float("nan"),
        "distance_RMSE_m":     round(float(np.sqrt(np.mean(np.square(dist_errs)))), 4)
                               if dist_errs else float("nan"),
        "distance_n_matched":  len(dist_errs),

        # Direction accuracy
        "direction_zone_MAE":  round(float(np.mean(dir_errs)), 4) if dir_errs else float("nan"),

        # Fields evaluate.py has but baseline cannot compute
        "mAP_50":              float("nan"),
        "top3_nearest_recall": float("nan"),
        "obstacle_recall":     float("nan"),

        "per_image": per_image,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Markdown report
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(
    all_results: Dict[str, List[Dict]],
    f_pixels: float,
    save_path: str,
) -> None:
    lines = [
        "# Baseline Evaluation Report",
        "",
        "**Method**: Triangle theorem  —  D = (f × H_real) / H_pixel",
        f"**Focal length**: {f_pixels:.1f} px (auto-read from EXIF)",
        "**Direction**: left / center / right  *(matches scene_analyzer.py)*",
        "**Alert format**: `{direction} {dist}m {class}`  *(matches proximity_alerter.py)*",
        "**Distance key**: `depth_stats.top_k_100`  *(matches evaluate.py)*",
        "",
        "---", "",
        "## Per-Object Results", "",
        "| Image | Class | GT dist | Pred dist | Err (m) | GT dir | Pred dir | Dir err | Alert |",
        "|-------|-------|---------|-----------|---------|--------|----------|---------|-------|",
    ]

    dist_errs, dir_errs = [], []
    for img_name, obj_list in all_results.items():
        for obj in obj_list:
            if not obj["detected"]:
                lines.append(
                    f"| {img_name} | {obj['class']} | {obj['gt_d']} | — | — | "
                    f"{obj['gt_direction']} | — | — | ❌ |"
                )
                continue
            lines.append(
                f"| {img_name} | {obj['class']} | {obj['gt_d']} | {obj['pred_d']} | "
                f"{obj['acc_dist']} | {obj['gt_direction']} | {obj['direction']} | "
                f"{obj['dir_error']} | {obj['alert']} |"
            )
            if obj["acc_dist"] is not None: dist_errs.append(obj["acc_dist"])
            if obj["dir_error"] is not None and obj["dir_error"] >= 0:
                dir_errs.append(obj["dir_error"])

    lines += ["", "---", "", "## Summary", ""]
    if dist_errs:
        lines.append(f"- **Distance MAE**:  {np.mean(dist_errs):.3f} m")
        lines.append(f"- **Distance RMSE**: {np.sqrt(np.mean(np.square(dist_errs))):.3f} m")
    if dir_errs:
        lines.append(f"- **Direction zone MAE**: {np.mean(dir_errs):.2f} "
                     f"(0=exact, 1=adjacent, 2=opposite)")
    n_total    = sum(len(v) for v in all_results.values())
    n_detected = sum(1 for v in all_results.values() for o in v if o["detected"])
    lines.append(f"- **Detection rate**: {n_detected}/{n_total} "
                 f"({100*n_detected/max(n_total,1):.0f}%)")

    lines += [
        "", "---", "",
        "## Comparing with Full Pipeline", "",
        "Both files share the same JSON keys:", "",
        "```python",
        "import json",
        'b = json.load(open("baseline_results.json"))',
        'f = json.load(open("results/metrics/hazard_eval.json"))',
        'print(\"Baseline  dist MAE:\", b[\"distance_MAE_m\"])',
        'print(\"Full pipe dist MAE:\", f[\"distance_MAE_m\"])',
        "```",
        "", "## Baseline Limitations", "",
        "1. Requires object in YOLO-80 — fire, smoke, wet floor undetectable.",
        "2. Single H_real per class — error grows for partial/non-standard objects.",
        "3. Triangle theorem assumes object is upright.",
        "4. No per-pixel depth — DepthAnything's mask-aware top_k_100 is more robust.",
    ]

    Path(save_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir",   default=".")
    p.add_argument("--gt",           default="gt_labels_sample.json")
    p.add_argument("--output",       default="baseline_results.json")
    p.add_argument("--report",       default="baseline_report.md")
    p.add_argument("--scale-factor", type=int,   default=1)
    p.add_argument("--pixel-bbox",   action="store_true")
    p.add_argument("--img-size",     type=int, nargs=2, default=[4284, 5712])
    p.add_argument("--focal-px",     type=float, default=None)
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
    gt_data = {k: v for k, v in gt_data.items() if not k.startswith("_")}

    print(f"\n{'='*55}")
    print("  YOLO TRIANGLE-THEOREM BASELINE")
    print(f"  Schema aligned with evaluate.py")
    print(f"{'='*55}")
    print(f"  GT: {gt_path.name}  ({len(gt_data)} images)")

    first_img = next(
        (images_dir / n for n in gt_data if (images_dir / n).is_file()), None
    )
    if args.focal_px:
        f_global, img_w_global = args.focal_px, args.img_size[0]
    elif first_img:
        f_global, img_w_global, _ = focal_length_from_exif(str(first_img))
    else:
        print("No images found.")
        return 1

    print(f"  Focal len : {f_global:.1f} px")
    print(f"\nLoading Yolo26 ...")
    yolo = Yolo26(scale_factor=args.scale_factor)

    all_results: Dict[str, List[Dict]] = {}

    for img_name, gt_objects in gt_data.items():
        img_path = images_dir / img_name
        if not img_path.is_file():
            print(f"  ⚠ Not found: {img_path}")
            continue

        try:
            f_px, img_w, _ = focal_length_from_exif(str(img_path))
        except Exception:
            f_px, img_w = f_global, img_w_global

        if args.focal_px:
            f_px = args.focal_px

        print(f"\n{img_name}")
        obj_results = evaluate_image(str(img_path), gt_objects, yolo, f_px, img_w)

        for obj in obj_results:
            if not obj["detected"]:
                print(f"  ❌ {obj['class']:20s} NOT detected")
            else:
                print(f"  ✅ {obj['class']:20s}  "
                      f"gt={obj['gt_d']}m  pred={obj['pred_d']}m  Δd={obj['acc_dist']}m  "
                      f"gt_dir={obj['gt_direction']}  pred_dir={obj['direction']}  "
                      f"alert='{obj['alert']}'")

        all_results[img_name] = obj_results

    summary = build_summary_json(all_results, f_global, str(gt_path))

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  JSON : {args.output}")

    generate_report(all_results, f_global, args.report)

    print(f"\n{'='*55}")
    print(f"  Distance MAE : {summary['distance_MAE_m']:.3f} m")
    print(f"  Distance RMSE: {summary['distance_RMSE_m']:.3f} m")
    print(f"  Dir zone MAE : {summary['direction_zone_MAE']:.2f}")
    print(f"  Detected     : {summary['distance_n_matched']} / "
          f"{sum(len(v) for v in all_results.values())}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
