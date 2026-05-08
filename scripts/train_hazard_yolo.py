#!/usr/bin/env python3
"""Fine-tune YOLO on the HK + Roboflow indoor dataset.

Picks the best available device automatically (CUDA → MPS → CPU).

Usage:
    python3 scripts/train_hazard_yolo.py --epochs 50 --imgsz 640 --batch 16
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

from convert_to_yolo import HAZARD_CLASSES  # noqa: E402


def pick_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data",
                        default=os.path.join(config.DATA_DIR, "HK_custom_for_finetuning", "data.yaml"))
    parser.add_argument("--weights", default=config.YOLO_MODEL_PATH,
                        help="Starting weights (filename → ultralytics auto-downloads).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default="hazard_baseline")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    from ultralytics import YOLO

    device = args.device or pick_device()
    logger.info(f"Device:  {device}")
    logger.info(f"Weights: {args.weights}")
    logger.info(f"Data:    {args.data}")
    logger.info(f"Classes: {HAZARD_CLASSES}")

    model = YOLO(args.weights)
    results = model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=device, name=args.name, lr0=args.lr0, patience=args.patience,
        cos_lr=True, warmup_epochs=3,
        mosaic=1.0, mixup=0.1, hsv_v=0.4, fliplr=0.5, scale=0.5, close_mosaic=10,
        plots=True,
    )
    logger.info(f"Best weights: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
