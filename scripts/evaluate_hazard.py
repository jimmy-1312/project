#!/usr/bin/env python3
"""
Evaluate a YOLO checkpoint on the HK indoor val/test set, computing both
detection metrics (Ultralytics built-in) and our custom alert-quality
metrics (distance MAE, top-K ranking accuracy, obstacle recall).

Designed to produce the comparison table for the final report:

    | model            | mAP50 | mAP50-95 | dist MAE | top3 acc | obs rec |
    |------------------|-------|----------|----------|----------|---------|
    | vanilla          |       |          |          |          |         |
    | A                |       |          |          |          |         |
    | A+C              |       |          |          |          |         |
    | A+B+C            |       |          |          |          |         |

Usage:
    python3 scripts/evaluate_hazard.py --weights runs/detect/baseline/weights/best.pt \\
        --tag A --output results/metrics/hazard_eval_A.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ============================================================
# IoU + matching
# ============================================================


def iou_xywh_normalized(a: np.ndarray, b: np.ndarray, img_w: int, img_h: int) -> float:
    """IoU between two YOLO-format boxes [cx, cy, w, h] in [0,1]."""
    ax1 = (a[0] - a[2] / 2) * img_w
    ay1 = (a[1] - a[3] / 2) * img_h
    ax2 = (a[0] + a[2] / 2) * img_w
    ay2 = (a[1] + a[3] / 2) * img_h
    bx1 = (b[0] - b[2] / 2) * img_w
    by1 = (b[1] - b[3] / 2) * img_h
    bx2 = (b[0] + b[2] / 2) * img_w
    by2 = (b[1] + b[3] / 2) * img_h
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return float(inter / union) if union > 0 else 0.0


def greedy_match(
    pred_boxes: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    img_w: int,
    img_h: int,
    iou_thresh: float = 0.5,
) -> List[Tuple[int, int]]:
    """
    Greedy 1-1 matching of predictions to ground truth above iou_thresh.
    Returns list of (pred_idx, gt_idx) pairs.
    """
    if not pred_boxes or not gt_boxes:
        return []
    iou_mat = np.array(
        [[iou_xywh_normalized(p, g, img_w, img_h) for g in gt_boxes] for p in pred_boxes]
    )
    pairs: List[Tuple[int, int]] = []
    used_pred = set()
    used_gt = set()
    while True:
        # Find global max iou not yet used
        best = -1.0
        bi = bj = -1
        for i in range(iou_mat.shape[0]):
            if i in used_pred:
                continue
            for j in range(iou_mat.shape[1]):
                if j in used_gt:
                    continue
                if iou_mat[i, j] > best:
                    best = iou_mat[i, j]
                    bi, bj = i, j
        if best < iou_thresh or bi < 0:
            break
        pairs.append((bi, bj))
        used_pred.add(bi)
        used_gt.add(bj)
    return pairs


# ============================================================
# Per-image evaluation
# ============================================================


def load_gt_for_image(label_path: Path, distances_for_stem: Optional[List[float]] = None) -> List[Dict]:
    """
    Read a 5-column YOLO label file (cls cx cy w h) and pair each row with
    a per-target distance from distances.json (passed in by the caller).

    Args:
        label_path: path to <stem>.txt
        distances_for_stem: list of distances aligned with label row order.
            None values are treated as NaN. If shorter than label count, the
            remainder is filled with NaN.

    Returns:
        list of {"cls": int, "bbox": [cx, cy, w, h], "distance_m": float}.
    """
    if not label_path.is_file():
        return []
    out = []
    distances_for_stem = distances_for_stem or []
    for i, raw in enumerate(label_path.read_text().splitlines()):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) < 5:
            continue
        d = distances_for_stem[i] if i < len(distances_for_stem) else None
        out.append(
            {
                "cls": int(float(parts[0])),
                "bbox": np.array([float(p) for p in parts[1:5]], dtype=np.float32),
                "distance_m": float("nan") if d is None else float(d),
            }
        )
    return out


def predict_with_distance(
    yolo_model,
    image_path: Path,
    conf: float,
    iou: float,
    depth_estimator=None,
) -> List[Dict]:
    """
    Run YOLO + (optionally) DepthEstimator on one image. Return predictions
    with bbox in YOLO normalized format and a distance_m where available.

    distance_m comes from depth_estimator (top_k_100 within bbox) when supplied,
    else NaN.
    """
    import cv2

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        from PIL import Image
        rgb = np.array(Image.open(image_path).convert("RGB"))
    else:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    # YOLO predictions
    res = yolo_model.predict(rgb, conf=conf, iou=iou, verbose=False)[0]
    boxes_xyxy = res.boxes.xyxy.cpu().numpy()  # absolute pixels
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()

    # Depth map (whole image)
    depth_map = None
    if depth_estimator is not None:
        depth_map = depth_estimator.estimate_depth(rgb)
        if depth_map.shape != (H, W):
            from PIL import Image as _Image
            depth_map = np.array(
                _Image.fromarray(depth_map.astype(np.float32)).resize((W, H), resample=_Image.BILINEAR),
                dtype=np.float32,
            )

    # Optional COCO → HK class remap (for vanilla baseline mode).
    coco_to_hk = getattr(predict_with_distance, "_coco_to_hk", None)

    out = []
    for i, (xyxy, c, p) in enumerate(zip(boxes_xyxy, cls_ids, confs)):
        if coco_to_hk is not None:
            mapped = coco_to_hk.get(int(c))
            if mapped is None:
                continue  # vanilla model predicted something not in our taxonomy → drop
            c = mapped
        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) / 2.0 / W
        cy = (y1 + y2) / 2.0 / H
        w = (x2 - x1) / W
        h = (y2 - y1) / H

        distance_m = float("nan")
        if depth_map is not None:
            xs1, ys1 = int(max(0, x1)), int(max(0, y1))
            xs2, ys2 = int(min(W, x2)), int(min(H, y2))
            if xs2 > xs1 and ys2 > ys1:
                patch = depth_map[ys1:ys2, xs1:xs2]
                patch = patch[np.isfinite(patch) & (patch > 0)]
                if patch.size > 0:
                    k = min(100, patch.size)
                    closest = np.partition(patch, k - 1)[:k]
                    distance_m = float(np.mean(closest))

        out.append(
            {
                "cls": int(c),
                "bbox": np.array([cx, cy, w, h], dtype=np.float32),
                "confidence": float(p),
                "distance_m": distance_m,
            }
        )
    return out


# ============================================================
# Custom metrics
# ============================================================


def topk_nearest_recall(
    preds: List[Dict],
    gts: List[Dict],
    img_w: int,
    img_h: int,
    k: int,
    match_iou: float = 0.3,
) -> float:
    """
    "Of the K closest ground-truth objects, how many did the detector find?"

    Concretely:
      1. Pick the K GT objects with smallest finite distance_m.
      2. For each, count it as recovered if ANY predicted box has IoU > match_iou.
      3. Return recovered / K.

    This is the metric that matters for the assistive use case: warning the user
    about the closest hazards. Returns NaN if no GT has a finite distance.
    """
    gt_with_dist = [g for g in gts if np.isfinite(g.get("distance_m", float("nan")))]
    if not gt_with_dist:
        return float("nan")

    # Sort by distance ascending; take K nearest
    gt_with_dist.sort(key=lambda g: g["distance_m"])
    topk_gt = gt_with_dist[:k]

    pred_boxes = [p["bbox"] for p in preds]
    if not pred_boxes:
        return 0.0

    recovered = 0
    for g in topk_gt:
        gt_box = g["bbox"]
        for pb in pred_boxes:
            if iou_xywh_normalized(pb, gt_box, img_w, img_h) > match_iou:
                recovered += 1
                break
    return recovered / max(len(topk_gt), 1)


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True,
                        help="Path to YOLO weights to evaluate (e.g. runs/detect/.../best.pt).")
    parser.add_argument("--data",
                        default=os.path.join(config.DATA_DIR, "HK_custom_for_finetuning", "data.yaml"),
                        help="Ultralytics dataset YAML.")
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="Which split to evaluate on.")
    parser.add_argument("--conf", type=float, default=0.10,
                        help="Confidence threshold for our custom-metric pass "
                             "(distance MAE / top-K ranking / obstacle recall). "
                             "NOT used for mAP — that uses ultralytics' default "
                             "(0.001) so the precision-recall curve covers all "
                             "thresholds correctly.")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="NMS IoU threshold passed to YOLO predict. Not the matching threshold.")
    parser.add_argument("--match-iou", type=float, default=0.3,
                        help="IoU threshold for matching predictions to GT in our custom "
                             "metric pass (distance MAE / top-K nearest recall / obstacle recall). "
                             "Lower than mAP's 0.5 because we care about approximate localization, "
                             "not tight bbox accuracy.")
    parser.add_argument("--top-k", type=int, default=3,
                        help="K for top-K nearest-GT recall. Default 3.")
    parser.add_argument("--no-depth", action="store_true",
                        help="Skip depth pipeline (distance MAE will be NaN).")
    parser.add_argument("--tag", default="model",
                        help="Tag for output JSON (e.g. 'A', 'A+C', 'baseline').")
    parser.add_argument("--output",
                        default=os.path.join(config.METRICS_DIR, "hazard_eval.json"),
                        help="JSON output path.")
    parser.add_argument("--vanilla", action="store_true",
                        help="Vanilla baseline: predict with COCO 80 classes and remap to "
                             "our 8-class taxonomy. Use with COCO-pretrained YOLO weights "
                             "(e.g. yolo11s.pt). Classes without a COCO equivalent "
                             "(door, obstacle) get recall=0 by construction.")
    args = parser.parse_args()

    from ultralytics import YOLO  # lazy
    yolo_model = YOLO(args.weights)

    # Vanilla mode: remap COCO predict ids → our HK class ids for the custom-metric pass.
    if args.vanilla:
        # COCO id → our HK id. Classes with no COCO equivalent are simply absent → recall=0.
        coco_to_hk = {
            56: 0,  # chair → chair
            60: 1,  # dining table → table        (closest match)
            72: 2,  # refrigerator → refrigerator
            59: 4,  # bed → bed
            57: 5,  # couch → couch
            # 60 also re-maps to 6 (dining_table) — we keep first mapping; report explicitly.
            # door (3) and obstacle (7): no COCO equivalent — model can't predict them.
        }
        predict_with_distance._coco_to_hk = coco_to_hk
        logger.info(f"Vanilla mode: remapping {len(coco_to_hk)} COCO classes → HK taxonomy. "
                    f"door/obstacle/dining_table will have recall=0 (no COCO equivalent / "
                    f"merged into table).")

    # Ultralytics built-in val gives us mAP. Do NOT pass `conf` here —
    # mAP is integrated over the full PR curve, so ultralytics uses 0.001
    # internally and overriding with a high threshold collapses mAP to 0.
    logger.info("Running ultralytics val (mAP)...")
    metrics = yolo_model.val(data=args.data, split=args.split, iou=args.iou,
                             plots=False, verbose=False)
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)

    # Per-class mAP
    per_class = {}
    for i, name in enumerate(metrics.names.values() if isinstance(metrics.names, dict)
                             else metrics.names):
        try:
            per_class[name] = float(metrics.box.maps[i])
        except (IndexError, AttributeError):
            pass

    # Custom metrics — distance MAE, top-K ranking, obstacle recall
    depth_estimator = None
    if not args.no_depth:
        from src.depth_estimator import DepthEstimator
        depth_estimator = DepthEstimator()

    # Locate split images + labels
    from PIL import Image
    data_root = Path(args.data).parent
    img_dir = data_root / "images" / args.split
    lbl_dir = data_root / "labels" / args.split

    # Per-target GT distances live out-of-band in distances.json (we use 5-col labels).
    distances_by_stem = {}
    dist_json = data_root / "distances.json"
    if dist_json.is_file():
        blob = json.loads(dist_json.read_text())
        raw = blob.get(args.split, {}) or {}
        distances_by_stem = {
            stem: [float("nan") if d is None else float(d) for d in dists]
            for stem, dists in raw.items()
        }

    images = sorted(p for p in img_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    logger.info(f"Custom-metric pass over {len(images)} {args.split} images...")

    dist_errors: List[float] = []
    topk_scores: List[float] = []
    obstacle_total = 0
    obstacle_recalled = 0
    OBSTACLE_CLS = 7

    per_image_records: List[Dict] = []

    for img_path in images:
        gts = load_gt_for_image(
            lbl_dir / f"{img_path.stem}.txt",
            distances_for_stem=distances_by_stem.get(img_path.stem),
        )
        if not gts:
            continue

        with Image.open(img_path) as im:
            W, H = im.size

        preds = predict_with_distance(yolo_model, img_path,
                                       conf=args.conf, iou=args.iou,
                                       depth_estimator=depth_estimator)

        # IoU match preds → gts (class-agnostic). Use a separate, looser threshold
        # for matching: mAP@0.5 wants tight boxes, but for distance / nearest-recall
        # we care about "did the model find approximately the right region".
        pred_boxes = [p["bbox"] for p in preds]
        gt_boxes = [g["bbox"] for g in gts]
        pairs = greedy_match(pred_boxes, gt_boxes, W, H, iou_thresh=args.match_iou)

        # Distance MAE on matched pairs that have BOTH finite distances.
        # GT distance is finite only for HK images (Roboflow rows have NaN).
        for pi, gi in pairs:
            gd = gts[gi]["distance_m"]
            pd = preds[pi]["distance_m"]
            if np.isfinite(gd) and np.isfinite(pd):
                dist_errors.append(abs(pd - gd))

        # Top-K nearest recall: of the K closest GT objects, how many were detected?
        topk_recall = topk_nearest_recall(
            preds, gts, img_w=W, img_h=H, k=args.top_k, match_iou=args.match_iou
        )
        if np.isfinite(topk_recall):
            topk_scores.append(topk_recall)

        # Obstacle recall — was there a "obstacle" prediction matched to a GT obstacle?
        for gi, g in enumerate(gts):
            if g["cls"] != OBSTACLE_CLS:
                continue
            obstacle_total += 1
            for pi, gj in pairs:
                if gj == gi and preds[pi]["cls"] == OBSTACLE_CLS:
                    obstacle_recalled += 1
                    break

        per_image_records.append({
            "image": img_path.name,
            "n_gt": len(gts),
            "n_pred": len(preds),
            "n_matched": len(pairs),
            "topk_recall": float(topk_recall) if np.isfinite(topk_recall) else None,
        })

    topk_key = f"top{args.top_k}_nearest_recall"
    summary = {
        "tag": args.tag,
        "weights": str(args.weights),
        "split": args.split,
        "conf": args.conf,
        "iou_nms": args.iou,
        "match_iou": args.match_iou,
        "top_k": args.top_k,
        "mAP_50": map50,
        "mAP_50_95": map5095,
        "per_class_mAP_50_95": per_class,
        "distance_MAE_m": float(np.mean(dist_errors)) if dist_errors else float("nan"),
        "distance_n_matched": len(dist_errors),
        topk_key: float(np.mean(topk_scores)) if topk_scores else float("nan"),
        "obstacle_recall": (obstacle_recalled / obstacle_total) if obstacle_total else float("nan"),
        "obstacle_total": obstacle_total,
        "obstacle_recalled": obstacle_recalled,
        "per_image": per_image_records,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info(f"[{args.tag}] @ {args.split}")
    logger.info(f"  mAP@0.5             : {map50:.4f}")
    logger.info(f"  mAP@0.5:0.95        : {map5095:.4f}")
    logger.info(f"  Distance MAE        : "
                f"{summary['distance_MAE_m']:.3f} m  ({summary['distance_n_matched']} pairs)")
    logger.info(f"  Top-{args.top_k} nearest recall: {summary[topk_key]:.3f}")
    logger.info(f"  Obstacle recall     : {summary['obstacle_recall']}  "
                f"({obstacle_recalled}/{obstacle_total})")
    logger.info(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
