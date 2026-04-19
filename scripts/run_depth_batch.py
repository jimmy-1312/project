"""
Batch depth estimation runner for a user-supplied image folder.

Loads every image in a directory, runs Depth Anything V2 (Small) once per image
(model loaded only once), and saves:
  - {stem}_depth.png      : side-by-side visualization (input | depth colormap)
  - {stem}_depth_raw.png  : 16-bit raw depth (normalized 0-1)
  - {stem}_stats.json     : per-image qualitative statistics
  - _summary.json         : aggregated stats + per-image table

Usage:
    python scripts/run_depth_batch.py --images-dir data/HK_custom_for_finetuning
    python scripts/run_depth_batch.py --images-dir ~/photos/kitchen --output-dir results/depth_maps/kitchen
    python scripts/run_depth_batch.py --images-dir some_folder --dry-run     # no model needed
    python scripts/run_depth_batch.py --images-dir some_folder --recursive   # descend into subfolders
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add project root to path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src.evaluation import evaluate_depth_qualitative  # noqa: E402


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_images(root: Path, recursive: bool) -> List[Path]:
    """Collect image paths, sorted, unique."""
    globber = root.rglob if recursive else root.glob
    files = [p for p in globber("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(set(files))


def load_image_rgb(path: Path) -> np.ndarray:
    """Load image as (H, W, 3) uint8 RGB."""
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"Could not decode image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def save_side_by_side(
    image: np.ndarray,
    depth: np.ndarray,
    out_path: Path,
    title: str = "",
) -> None:
    """Save input + depth colormap as a single PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")

    im = axes[1].imshow(depth, cmap="inferno")
    axes[1].set_title("Depth (relative, inferno)")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_depth_raw_16bit(depth: np.ndarray, out_path: Path) -> None:
    """Save a 16-bit PNG of depth normalized to [0, 1]."""
    depth_clipped = np.clip(depth, 0.0, 1.0)
    depth_uint16 = (depth_clipped * 65535.0).astype(np.uint16)
    cv2.imwrite(str(out_path), depth_uint16)


def make_dummy_depth(image: np.ndarray) -> np.ndarray:
    """Dry-run fallback: smooth gradient so the pipeline can be exercised."""
    h, w = image.shape[:2]
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    return 0.5 * y + 0.5 * x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Depth Anything V2 on every image in a folder.",
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Folder containing images (jpg/png/bmp/tif/webp).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: results/depth_maps/<images-dir basename>",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Descend into subfolders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip model load; use synthetic depth (CI/structure check).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    images_dir = Path(args.images_dir).expanduser().resolve()
    if not images_dir.is_dir():
        print(f"[ERROR] Not a directory: {images_dir}", file=sys.stderr)
        return 2

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path(config.RESULTS_DIR) / "depth_maps" / images_dir.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = find_images(images_dir, args.recursive)
    if not image_paths:
        print(f"[ERROR] No supported images found under {images_dir}", file=sys.stderr)
        return 2

    print(f"[1/3] Found {len(image_paths)} image(s) under {images_dir}")
    print(f"      Output: {out_dir}")

    estimator = None
    if not args.dry_run:
        print("[2/3] Loading DepthEstimator (first run downloads ~100 MB)...")
        from src.depth_estimator import DepthEstimator  # lazy import

        estimator = DepthEstimator()
    else:
        print("[2/3] --dry-run: skipping model load")

    per_image = []
    skipped = []
    total_time = 0.0

    print(f"[3/3] Running depth inference on {len(image_paths)} image(s)")
    for path in tqdm(image_paths, unit="img"):
        try:
            image = load_image_rgb(path)
        except (ValueError, cv2.error) as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
            continue

        if args.dry_run:
            depth = make_dummy_depth(image)
            inference_time_s = 0.0
        else:
            t0 = time.perf_counter()
            depth = estimator.estimate_depth(image)
            inference_time_s = time.perf_counter() - t0
            total_time += inference_time_s

        stats = evaluate_depth_qualitative(depth)
        stats["inference_time_s"] = float(inference_time_s)
        stats["image_shape"] = list(image.shape)
        stats["image_path"] = str(path)

        stem = path.stem
        save_side_by_side(
            image,
            depth,
            out_dir / f"{stem}_depth.png",
            title=f"{stem}  ({image.shape[1]}x{image.shape[0]})",
        )
        save_depth_raw_16bit(depth, out_dir / f"{stem}_depth_raw.png")
        with open(out_dir / f"{stem}_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        per_image.append(stats)

    summary = {
        "images_dir": str(images_dir),
        "output_dir": str(out_dir),
        "recursive": bool(args.recursive),
        "dry_run": bool(args.dry_run),
        "num_processed": len(per_image),
        "num_skipped": len(skipped),
        "total_inference_time_s": float(total_time),
        "avg_inference_time_s": float(total_time / len(per_image)) if per_image else 0.0,
        "per_image": per_image,
        "skipped": skipped,
    }
    with open(out_dir / "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    print(f"  Processed : {summary['num_processed']}")
    print(f"  Skipped   : {summary['num_skipped']}")
    if not args.dry_run and per_image:
        print(f"  Total time: {total_time:.2f}s  (avg {summary['avg_inference_time_s']:.2f}s / img)")
    print(f"  Summary   : {out_dir / '_summary.json'}")

    if skipped:
        print("\n  Skipped files:")
        for s in skipped:
            print(f"    - {s['path']}: {s['reason']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
