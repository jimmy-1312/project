"""HazardSight pipeline modules.

  - detector.py            YOLO11s wrapper
  - segmentor.py           MobileSAM wrapper
  - depth_estimator.py     Depth Anything V2 wrapper (metric-indoor)
  - scene_analyzer.py      runs all 3 above and aggregates per-object results
  - obstacle_proposer.py   depth-only fallback for objects YOLO misses
  - proximity_alerter.py   ranks results by distance and emits text alerts

Heavy imports (ultralytics, transformers, torch) are deferred to constructors,
so importing this package is cheap.
"""
