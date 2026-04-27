#!/usr/bin/env python3
"""
NYU Depth v2 Evaluation for our DepthEstimator.

Computes the standard monocular depth metrics (AbsRel, RMSE, RMSE_log, δ1, δ2, δ3)
on NYU Depth v2 to validate that our depth backbone produces accurate metric depth.

This is the **quantitative evaluation** referenced in the final report's
Experiments section. Run it once to populate the depth-accuracy table.

Two ways to feed data:

  (A) HuggingFace dataset (recommended — no manual download)
        python3 scripts/evaluate_depth_nyu.py \\
            --hf-dataset sayakpaul/nyu_depth_v2 --split validation --max-images 200

  (B) Local RGB + GT directory (matching filenames, .jpg + .npy or 16-bit .png)
        python3 scripts/evaluate_depth_nyu.py \\
            --rgb-dir  /path/to/nyu/rgb \\
            --gt-dir   /path/to/nyu/depth \\
            --gt-scale 0.001     # if depth is uint16 mm → meters

Metric definitions (standard, e.g. Eigen et al. 2014):
    AbsRel    = mean(|d_pred - d_gt| / d_gt)
    RMSE      = sqrt(mean((d_pred - d_gt)^2))
    RMSE_log  = sqrt(mean((log d_pred - log d_gt)^2))
    δ_t       = fraction with max(d_pred/d_gt, d_gt/d_pred) < 1.25^t  (t=1,2,3)

Eval crop:
    NYU's standard "Eigen crop" = [45:471, 41:601] for 480x640 images,
    used by every depth paper for fair comparison. Applied automatically
    when image is exactly 480x640; otherwise full image is used.

Valid pixel mask:
    Pixels with GT depth in (eval_min_m, eval_max_m] (default 0.001 - 10 m)
    are evaluated. Zero/inf/NaN GT pixels are excluded.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

# Project root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ============================================================
# Metrics
# ============================================================


def compute_depth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    eval_min: float = 1e-3,
    eval_max: float = 10.0,
) -> Dict[str, float]:
    """
    Standard monocular depth metrics over the valid GT mask.

    Args:
        pred: predicted depth (H, W) in meters
        gt:   ground-truth depth (H, W) in meters
        eval_min, eval_max: clamp range for valid GT pixels (meters)

    Returns:
        Dict of {abs_rel, rmse, rmse_log, delta1, delta2, delta3, n_valid}.
        Empty dict-like with NaNs if no valid pixels.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    valid = np.isfinite(gt) & np.isfinite(pred) & (gt > eval_min) & (gt <= eval_max)
    if not valid.any():
        return {k: float("nan") for k in
                ["abs_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3"]} | {"n_valid": 0}

    p = np.clip(pred[valid], eval_min, None)
    g = gt[valid]

    abs_rel = float(np.mean(np.abs(p - g) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    rmse_log = float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2)))

    ratio = np.maximum(p / g, g / p)
    delta1 = float(np.mean(ratio < 1.25))
    delta2 = float(np.mean(ratio < 1.25 ** 2))
    delta3 = float(np.mean(ratio < 1.25 ** 3))

    return {
        "abs_rel": abs_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
        "n_valid": int(valid.sum()),
    }


# ============================================================
# Optional median-scale alignment (for non-metric models)
# ============================================================


def align_with_median_scale(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Scale `pred` so that median(pred[valid]) matches median(gt[valid]).
    Used when evaluating a relative-depth model — gives the "best case" scaled
    prediction. For a true metric model this is unnecessary (and unfair to skip
    on baselines), so this is gated behind --align-median.
    """
    p = pred[valid]
    g = gt[valid]
    if p.size == 0 or g.size == 0:
        return pred
    scale = float(np.median(g) / np.maximum(np.median(p), 1e-8))
    return pred * scale


# ============================================================
# Eigen crop
# ============================================================


def apply_eigen_crop(arr: np.ndarray) -> np.ndarray:
    """
    NYU standard Eigen crop. Only applied if shape matches the standard
    NYU resolution; otherwise returned unchanged.
    """
    if arr.shape[:2] == (480, 640):
        return arr[45:471, 41:601]
    return arr


# ============================================================
# Data loaders
# ============================================================


def iter_local_pairs(
    rgb_dir: str,
    gt_dir: str,
    gt_scale: float,
    max_images: int,
) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Yield (name, rgb_uint8, gt_depth_meters) from local directory.

    Matches by stem: rgb/foo.jpg ↔ gt/foo.npy or gt/foo.png.
    """
    rgb_paths = sorted(
        p for p in Path(rgb_dir).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if max_images > 0:
        rgb_paths = rgb_paths[:max_images]

    for rgb_path in rgb_paths:
        stem = rgb_path.stem
        gt_npy = Path(gt_dir) / f"{stem}.npy"
        gt_png = Path(gt_dir) / f"{stem}.png"

        if gt_npy.exists():
            gt = np.load(gt_npy).astype(np.float32)
        elif gt_png.exists():
            gt = np.array(Image.open(gt_png), dtype=np.float32) * gt_scale
        else:
            logger.warning(f"  No GT for {rgb_path.name}, skipping.")
            continue

        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        yield stem, rgb, gt


def iter_hf_dataset(
    name: str,
    split: str,
    max_images: int,
) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Yield (name, rgb_uint8, gt_depth_meters) from a HuggingFace dataset.

    Expects entries with at least an 'image' field and a 'depth_map' field
    (matches sayakpaul/nyu_depth_v2 schema). Other schemas: tweak below.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "pip3 install datasets   # required for --hf-dataset"
        ) from e

    ds = load_dataset(name, split=split, streaming=False)
    if max_images > 0:
        ds = ds.select(range(min(max_images, len(ds))))

    for i, item in enumerate(ds):
        # Field names vary; try common ones
        rgb_field = item.get("image") or item.get("rgb") or item.get("input_image")
        gt_field = item.get("depth_map") or item.get("depth") or item.get("gt_depth")
        if rgb_field is None or gt_field is None:
            logger.warning(f"  Item {i}: missing image/depth fields, skipping. Keys={list(item.keys())}")
            continue

        rgb = np.array(rgb_field.convert("RGB")) if hasattr(rgb_field, "convert") \
              else np.array(rgb_field)
        if hasattr(gt_field, "convert"):
            gt = np.array(gt_field, dtype=np.float32)
        else:
            gt = np.asarray(gt_field, dtype=np.float32)

        yield f"{i:06d}", rgb, gt


# ============================================================
# Evaluation
# ============================================================


def evaluate(
    estimator,
    pairs: Iterator[Tuple[str, np.ndarray, np.ndarray]],
    align_median: bool,
    use_eigen_crop: bool,
    eval_min: float,
    eval_max: float,
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Run estimator on each pair and accumulate metrics.

    Returns:
        per_image: list of {name, **metrics}
        aggregate: dict of mean metrics over all images
    """
    per_image = []
    accum = {k: [] for k in ["abs_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3"]}

    for i, (name, rgb, gt) in enumerate(pairs, 1):
        # Estimate depth
        pred = estimator.estimate_depth(rgb)
        # Resize pred to GT size if needed
        if pred.shape != gt.shape:
            from PIL import Image as _Image
            pred = np.array(
                _Image.fromarray(pred.astype(np.float32)).resize(
                    (gt.shape[1], gt.shape[0]), resample=_Image.BILINEAR
                )
            ).astype(np.float32)

        # Optional median-scale alignment (for non-metric models)
        if align_median:
            valid_mask = (gt > eval_min) & (gt <= eval_max) & np.isfinite(gt) & np.isfinite(pred)
            pred = align_with_median_scale(pred, gt, valid_mask)

        # Eigen crop
        if use_eigen_crop:
            pred_e = apply_eigen_crop(pred)
            gt_e = apply_eigen_crop(gt)
        else:
            pred_e, gt_e = pred, gt

        m = compute_depth_metrics(pred_e, gt_e, eval_min=eval_min, eval_max=eval_max)
        m["name"] = name
        per_image.append(m)
        for k in accum:
            if np.isfinite(m[k]):
                accum[k].append(m[k])

        if i % 10 == 0 or i == 1:
            logger.info(
                f"  [{i}] {name}: AbsRel={m['abs_rel']:.3f}  "
                f"RMSE={m['rmse']:.3f}m  δ1={m['delta1']:.3f}"
            )

    aggregate = {k: float(np.mean(v)) if v else float("nan") for k, v in accum.items()}
    aggregate["n_images"] = len(per_image)
    return per_image, aggregate


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf-dataset", type=str,
                     help="HuggingFace dataset id, e.g. 'sayakpaul/nyu_depth_v2'")
    src.add_argument("--rgb-dir", type=str, help="Directory of RGB images")

    parser.add_argument("--gt-dir", type=str,
                        help="(with --rgb-dir) Directory of GT depth (.npy or .png)")
    parser.add_argument("--gt-scale", type=float, default=0.001,
                        help="(with --rgb-dir + .png GT) factor to convert raw to meters. "
                             "Default 0.001 (uint16 mm → m).")
    parser.add_argument("--split", default="validation",
                        help="(with --hf-dataset) split name. Default: validation")

    parser.add_argument("--max-images", type=int, default=200,
                        help="Max images to evaluate (0 = all). Default 200 for fast iteration.")
    parser.add_argument("--align-median", action="store_true",
                        help="Median-scale align prediction to GT before metrics. "
                             "Use ONLY when evaluating a relative-depth model.")
    parser.add_argument("--no-eigen-crop", action="store_true",
                        help="Disable the standard NYU Eigen crop.")
    parser.add_argument("--eval-min", type=float, default=1e-3, help="Min valid GT depth (m).")
    parser.add_argument("--eval-max", type=float, default=10.0, help="Max valid GT depth (m).")
    parser.add_argument("--output", default="results/metrics/depth_nyu.json",
                        help="Where to save JSON results.")
    parser.add_argument("--model", type=str, default=None,
                        help="Override config.DEPTH_MODEL_NAME for this run.")
    args = parser.parse_args()

    # Validate
    if args.rgb_dir and not args.gt_dir:
        parser.error("--rgb-dir requires --gt-dir")

    # Load estimator
    from src.depth_estimator import DepthEstimator
    estimator = DepthEstimator(model_name=args.model)
    logger.info(f"Model: {estimator.model_name}")
    logger.info(f"Device: {estimator.device}")

    # Build data iterator
    if args.hf_dataset:
        pairs = iter_hf_dataset(args.hf_dataset, args.split, args.max_images)
        source = f"hf:{args.hf_dataset}/{args.split}"
    else:
        pairs = iter_local_pairs(args.rgb_dir, args.gt_dir, args.gt_scale, args.max_images)
        source = f"local:{args.rgb_dir}"

    logger.info(f"Source: {source}")
    logger.info(f"Eigen crop: {not args.no_eigen_crop}, align_median: {args.align_median}")
    logger.info(f"Valid GT range: ({args.eval_min}, {args.eval_max}] meters\n")

    # Evaluate
    per_image, aggregate = evaluate(
        estimator,
        pairs,
        align_median=args.align_median,
        use_eigen_crop=not args.no_eigen_crop,
        eval_min=args.eval_min,
        eval_max=args.eval_max,
    )

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out = {
        "model": estimator.model_name,
        "source": source,
        "max_images": args.max_images,
        "align_median": args.align_median,
        "eigen_crop": not args.no_eigen_crop,
        "eval_min": args.eval_min,
        "eval_max": args.eval_max,
        "aggregate": aggregate,
        "per_image": per_image,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info(f"Results over {aggregate['n_images']} images:")
    logger.info(f"  AbsRel    : {aggregate['abs_rel']:.4f}")
    logger.info(f"  RMSE      : {aggregate['rmse']:.4f} m")
    logger.info(f"  RMSE_log  : {aggregate['rmse_log']:.4f}")
    logger.info(f"  δ < 1.25  : {aggregate['delta1']:.4f}")
    logger.info(f"  δ < 1.25² : {aggregate['delta2']:.4f}")
    logger.info(f"  δ < 1.25³ : {aggregate['delta3']:.4f}")
    logger.info("=" * 60)
    logger.info(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
