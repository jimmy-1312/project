#!/usr/bin/env python3
"""Fine-tune YOLO on indoor data.

Two modes:

  Single training (default):
      python3 scripts/train_hazard_yolo.py --epochs 50 --imgsz 640 --batch 16

  K-fold cross-validation on Roboflow (per docs/PLAN_FINAL.md Stage 2):
      python3 scripts/train_hazard_yolo.py --kfold --epochs 10 --freeze 10 --lr0 5e-4

In k-fold mode, the script reads kfold_splits.json (made by restratify_splits.py),
trains one run per fold, and writes per-fold results under runs/detect/<name>_fold{k}/.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

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


def write_fold_yaml(yaml_path: Path, root: Path, train_stems: list, val_stems: list,
                    classes: list) -> None:
    """Write a temporary data.yaml that uses image-list .txt files for one fold."""
    train_txt = yaml_path.with_suffix(".train.txt")
    val_txt = yaml_path.with_suffix(".val.txt")
    img_dir = root / "images" / "roboflow"

    def list_for(stems):
        # ultralytics accepts absolute or root-relative paths; use absolute to be safe
        return [
            str(img_dir / fn)
            for s in stems
            for fn in os.listdir(img_dir)
            if Path(fn).stem == s
        ]

    train_paths = list_for(train_stems)
    val_paths = list_for(val_stems)
    train_txt.write_text("\n".join(train_paths) + "\n")
    val_txt.write_text("\n".join(val_paths) + "\n")

    yaml_path.write_text(
        f"# auto-generated for k-fold\n"
        f"path: {root}\n"
        f"train: {train_txt}\n"
        f"val: {val_txt}\n"
        f"test: images/test\n\n"
        f"nc: {len(classes)}\n"
        "names:\n"
        + "\n".join(f"  {i}: {n}" for i, n in enumerate(classes))
        + "\n"
    )


def train_one(weights, data_yaml, *, epochs, imgsz, batch, lr0, patience, freeze,
              device, name, amp=False):
    from ultralytics import YOLO
    model = YOLO(weights)
    # amp=False: workaround for MPS + Task Aligned Assigner shape-mismatch bug
    return model.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
        device=device, name=name, lr0=lr0, patience=patience,
        freeze=freeze if freeze > 0 else None,
        amp=amp,
        cos_lr=True, warmup_epochs=2,
        mosaic=1.0, mixup=0.1, hsv_v=0.4, fliplr=0.5, scale=0.5, close_mosaic=5,
        plots=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data",
                        default=os.path.join(config.DATA_DIR, "HK_custom_for_finetuning", "data.yaml"))
    parser.add_argument("--weights", default=config.YOLO_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default="hazard")
    parser.add_argument("--lr0", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--freeze", type=int, default=0,
                        help="Freeze first N layers (0 = no freeze; use 10 for backbone-only).")
    parser.add_argument("--kfold", action="store_true",
                        help="Run k-fold CV over Roboflow per kfold_splits.json.")
    args = parser.parse_args()

    device = args.device or pick_device()
    data_root = Path(args.data).parent

    if not args.kfold:
        logger.info(f"[single] device={device} weights={args.weights} freeze={args.freeze}")
        results = train_one(args.weights, args.data,
                            epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                            lr0=args.lr0, patience=args.patience, freeze=args.freeze,
                            device=device, name=args.name)
        logger.info(f"Best weights: {results.save_dir}/weights/best.pt")
        return

    # K-fold mode
    splits_path = data_root / "kfold_splits.json"
    if not splits_path.is_file():
        logger.error(f"{splits_path} not found. Run scripts/restratify_splits.py first.")
        sys.exit(1)
    splits = json.loads(splits_path.read_text())

    fold_summary = []
    for fold in splits["folds"]:
        k = fold["k"]
        logger.info("\n" + "=" * 60)
        logger.info(f"[fold {k}/{splits['n_folds']-1}] freeze={args.freeze} lr0={args.lr0}")
        yaml_k = data_root / f"_kfold_{k}.yaml"
        write_fold_yaml(yaml_k, data_root, fold["train"], fold["val"], HAZARD_CLASSES)

        try:
            results = train_one(args.weights, yaml_k,
                                epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                                lr0=args.lr0, patience=args.patience, freeze=args.freeze,
                                device=device, name=f"{args.name}_fold{k}")
            try:
                map50 = float(results.box.map50)
                map5095 = float(results.box.map)
            except Exception:
                map50 = map5095 = float("nan")
            fold_summary.append({"k": k, "save_dir": str(results.save_dir),
                                 "mAP50": map50, "mAP50-95": map5095})
            logger.info(f"[fold {k}] mAP@0.5={map50:.4f}  mAP@0.5:0.95={map5095:.4f}")
        except Exception as e:
            logger.error(f"[fold {k}] FAILED: {type(e).__name__}: {e}")
            fold_summary.append({"k": k, "save_dir": None,
                                 "mAP50": float("nan"), "mAP50-95": float("nan"),
                                 "error": f"{type(e).__name__}: {e}"})

        # Cleanup temp yaml + lists
        for f in (yaml_k, yaml_k.with_suffix(".train.txt"), yaml_k.with_suffix(".val.txt")):
            if f.is_file():
                f.unlink()

    # Aggregate
    logger.info("\n" + "=" * 60 + "\n[k-fold summary]")
    map50s = [s["mAP50"] for s in fold_summary if s["mAP50"] == s["mAP50"]]
    map5095s = [s["mAP50-95"] for s in fold_summary if s["mAP50-95"] == s["mAP50-95"]]
    if map50s:
        import statistics as st
        logger.info(f"mAP@0.5    mean={st.mean(map50s):.4f}  "
                    f"std={st.stdev(map50s) if len(map50s) > 1 else 0:.4f}")
        logger.info(f"mAP@0.5:.95 mean={st.mean(map5095s):.4f}  "
                    f"std={st.stdev(map5095s) if len(map5095s) > 1 else 0:.4f}")
    out = data_root.parent.parent / "results" / "metrics" / f"kfold_{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"folds": fold_summary, "lr0": args.lr0,
                               "freeze": args.freeze, "epochs": args.epochs}, indent=2))
    logger.info(f"Saved: {out}")


if __name__ == "__main__":
    main()
