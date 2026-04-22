"""
Download and verify model checkpoints.

Usage:
    python scripts/download_models.py

What it does:
  - Downloads the MobileSAM checkpoint to ``model/weights/mobile_sam.pt`` if
    it isn't already there.
  - Triggers an auto-download of YOLOv8m by instantiating the model.
  - Triggers an auto-download of Depth Anything V2 (Small) from HuggingFace.

Network notes:
  - All downloads pull from public CDNs (GitHub releases, HuggingFace Hub).
  - Run from the project root so relative paths resolve correctly.
"""

import os
import sys
import urllib.request
from pathlib import Path

# Allow running from the project root or from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402


# Mirror list for mobile_sam.pt — try in order until one works.
MOBILE_SAM_URLS = [
    # Original repo
    "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
    # Ultralytics CDN copy (used internally by `ultralytics`)
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/mobile_sam.pt",
]


def _download_with_progress(url: str, dest: Path) -> None:
    """Download `url` to `dest` with a simple stderr progress bar."""

    def _hook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100.0, downloaded * 100.0 / total_size)
        bar = "#" * int(pct / 2) + "-" * (50 - int(pct / 2))
        sys.stderr.write(
            f"\r  [{bar}] {pct:5.1f}%  ({downloaded / 1e6:.1f}/{total_size / 1e6:.1f} MB)"
        )
        sys.stderr.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    sys.stderr.write("\n")


def download_mobilesam() -> bool:
    """Download MobileSAM checkpoint to config.MOBILE_SAM_CHECKPOINT."""
    dest = Path(config.MOBILE_SAM_CHECKPOINT)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_file() and dest.stat().st_size > 1_000_000:
        print(f"[mobile_sam] Already present: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    for url in MOBILE_SAM_URLS:
        try:
            print(f"[mobile_sam] Trying {url}")
            _download_with_progress(url, dest)
            size_mb = dest.stat().st_size / 1e6
            if size_mb < 1.0:
                print(f"[mobile_sam] Suspiciously small file ({size_mb:.2f} MB); deleting")
                dest.unlink()
                continue
            print(f"[mobile_sam] OK: {dest} ({size_mb:.1f} MB)")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[mobile_sam] Failed: {e}")
            if dest.is_file():
                dest.unlink()

    print("[mobile_sam] All download URLs failed.")
    return False


def download_yolo() -> bool:
    """YOLO models auto-download from ultralytics on first instantiation."""
    try:
        from ultralytics import YOLO

        print(f"[yolo] Loading {config.YOLO_MODEL_NAME} (auto-downloads if needed)...")
        model = YOLO(config.YOLO_MODEL_NAME)
        n_classes = len(model.names)
        print(f"[yolo] OK: {n_classes} classes available")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[yolo] Failed: {e}")
        return False


def download_depth_anything() -> bool:
    """Depth Anything V2 auto-downloads from HuggingFace on first load."""
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        repo = config.DEPTH_PROCESSOR_NAME
        print(f"[depth] Loading processor and model from {repo}...")
        AutoImageProcessor.from_pretrained(repo)
        AutoModelForDepthEstimation.from_pretrained(repo)
        print("[depth] OK")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[depth] Failed: {e}")
        return False


def verify_all_models() -> None:
    """Print a status report for all expected model artifacts."""
    print("\n" + "=" * 60)
    print(" Verification report")
    print("=" * 60)

    sam_path = Path(config.MOBILE_SAM_CHECKPOINT)
    sam_ok = sam_path.is_file() and sam_path.stat().st_size > 1_000_000
    print(f"  MobileSAM checkpoint  : {'OK' if sam_ok else 'MISSING'}  ({sam_path})")

    yolo_ok = False
    try:
        from ultralytics import YOLO
        YOLO(config.YOLO_MODEL_NAME)
        yolo_ok = True
    except Exception:
        pass
    print(f"  YOLO weights          : {'OK' if yolo_ok else 'MISSING'}")

    depth_ok = False
    try:
        from transformers import AutoModelForDepthEstimation
        AutoModelForDepthEstimation.from_pretrained(config.DEPTH_PROCESSOR_NAME)
        depth_ok = True
    except Exception:
        pass
    print(f"  Depth Anything V2     : {'OK' if depth_ok else 'MISSING'}")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print(" Downloading model checkpoints")
    print("=" * 60)

    download_mobilesam()
    download_yolo()
    download_depth_anything()
    verify_all_models()

    print("\nModel setup complete.")
