# HazardSight

Indoor scene analyzer for assistive navigation. Given an image, returns the
top-K nearest objects with their distance and direction — designed to feed an
audio alert layer for visually impaired users.

## Pipeline (3 pre-trained models + thin glue)

```
image
  → YOLO11s         (object detection)
  → MobileSAM       (per-detection segmentation)
  → DepthAnythingV2 (per-pixel metric depth, indoor variant)
  → aggregation     (top_k_100 closest-pixels mean per mask)
  → proximity rank  (top-K by distance) + obstacle proposer (depth-only fallback)
  → alerts          ("left 1.5m chair")
```

## Layout

```
src/                          # 6 modules
  detector.py                 # YOLO wrapper
  segmentor.py                # MobileSAM wrapper
  depth_estimator.py          # DepthAnythingV2 wrapper
  scene_analyzer.py           # runs all 3 + aggregates per-object
  obstacle_proposer.py        # depth-only fallback for objects YOLO misses
  proximity_alerter.py        # rank by distance + natural-language alert

scripts/                      # 8 entry points (one role each)
  build_hk_dataset.py         # docx → unified labels JSON + raw images
  convert_to_yolo.py          # JSON → YOLO format (5-col labels + distances.json)
  merge_external_dataset.py   # mix in Roboflow-format public data
  restratify_splits.py        # re-shuffle train/val/test stratified by source
  train_hazard_yolo.py        # YOLO fine-tune
  evaluate_hazard.py          # mAP + Distance MAE + Top-K nearest recall
  evaluate_depth_nyu.py       # NYU Depth v2 depth-backbone metrics
  run_scene_analysis.py       # single-image / directory inference

tests/                        # 91 passing tests (network-free, mocked)
```

## Typical workflow

```bash
# 1) build dataset (one-time)
python3 scripts/build_hk_dataset.py
python3 scripts/convert_to_yolo.py
python3 scripts/merge_external_dataset.py --source <ROBOFLOW_EXPORT> \
    --dataset-name <NAME>
python3 scripts/restratify_splits.py

# 2) train
python3 scripts/train_hazard_yolo.py --epochs 50 --imgsz 640 --batch 16

# 3) evaluate
python3 scripts/evaluate_hazard.py --weights runs/detect/<run>/weights/best.pt \
    --conf 0.01 --tag clean --output results/metrics/eval_clean.json
python3 scripts/evaluate_depth_nyu.py --hf-dataset sayakpaul/nyu_depth_v2 \
    --max-images 200 --output results/metrics/depth_nyu.json

# 4) inference
python3 scripts/run_scene_analysis.py --image path/to/photo.jpg --nearest
```

## Models

- YOLO11s (Ultralytics) — fine-tuned on combined HK + Roboflow indoor data
- MobileSAM (Ultralytics) — pre-trained, frozen
- Depth Anything V2 Metric-Indoor (HuggingFace) — pre-trained, frozen

All weights download automatically on first run.
