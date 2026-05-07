# Baseline Evaluation Report

**Method**: Triangle theorem  —  D = (f × H_real) / H_pixel
**Focal length**: 2312.0 px (auto-read from EXIF)

---

## Per-Object Results

| Image | Class | gt_d (m) | pred_d (m) | acc_dist (m) | gt_clk | pred_clk | acc_direc |
|-------|-------|----------|------------|--------------|--------|----------|-----------|
| 1.jpg | refrigerator | 1.4 | 1.02 | 0.38 | 12 | 12 | 0 |

---

## Summary Statistics

- **Distance MAE**:  0.380 m
- **Distance RMSE**: 0.380 m
- **Distance max**:  0.380 m
- **Direction MAE**: 0.00 clock-hours
- **Direction max**: 0 clock-hours
- **YOLO detection rate**: 1/1 (100%)

---

## Baseline Limitations (addressed by improved model)

1. Requires object in YOLO-80 classes — fire, smoke, wet floor undetectable.
2. Single H_real per class — error grows if object is partial or non-standard size.
3. Triangle theorem assumes object is perpendicular — angled objects add error.
4. No depth sensor — Depth Anything addresses all three.