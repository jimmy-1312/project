# Depth-Aware YOLO Fine-Tuning — Project Plan

**Status:** draft for review (do not start implementation until approved)
**Owner:** Danny + collaborator + Claude
**Project context:** HK indoor assistive vision (visually-impaired user, top-K nearest objects + distance + direction alerts)

---

## 0. Decisions already locked in

| # | Decision | Value |
|---|---|---|
| 1 | Number of object classes | 8 (no merging) |
| 2 | `info` format | objects `[height_m, clock, distance_m]`; obstacles `[clock, distance_m]` |
| 3 | bbox format | normalized `[x1, y1, x2, y2]` ∈ [0, 1] for ALL sources (COMP4471 pixel coords get divided by 4284×5712) |
| 4 | Models to compare | vanilla YOLOv8 / **A** RGB fine-tune / **A+C** RGB + distance-weighted loss / **A+B+C** RGB-D + distance-weighted loss |
| 5 | Initial dataset | 28 images + augmentation only; add public data later if results too thin |

---

## 1. Class taxonomy (final)

8 object classes drawn from what's actually annotated in the 28-image set:

| id | name | source images that contain it |
|---|---|---|
| 0 | chair | many |
| 1 | table | several (we keep `dining table` annotations as `table` — see §1.1) |
| 2 | refrigerator | several |
| 3 | door | several |
| 4 | bed | comp4471 #1 |
| 5 | couch | comp4471 #2, #3 |
| 6 | dining table | comp4471 #6, separate from `table` (per user decision §0) |
| 7 | obstacle | env labels — wardrobe, wall, stairs, drying rack, doors not in YOLO80, etc. |

### 1.1 Note on table vs dining table
Decision §0 says 8 classes "no merging". The data has both `"table"` and `"dining table"` strings, and they refer to genuinely different things in our images (working desk vs eating table). We keep them separate as id=1 and id=6.

If during evaluation we see severe confusion between them (likely with only 28 images), we will revisit and merge in a follow-up — but the **initial** training keeps both. This is documented as Risk R3 in §10.

---

## 2. Data pipeline

### 2.1 Source files
- `uploads/sample.docx`         — 6 images, 6 labels
- `uploads/4471_data.docx`      — 16 images, 16 labels
- `uploads/COMP4471 Images.docx` — 6 images, 6 labels (high-res 4284×5712)

Each docx contains BOTH the inline JPEG/PNG image binaries AND a Python-dict-literal text block with two dicts: `gt_labels_objects` and `gt_labels_environment`.

### 2.2 Extraction script — `scripts/build_hk_dataset.py` (NEW)
Single script that runs end-to-end:

1. **Extract images**: unzip each docx, copy `word/media/*.{jpg,png}` to
   `data/HK_custom_for_finetuning/raw_images/{source}__{NN}.{ext}`.
2. **Parse labels**: regex-find each `gt_labels_*` block, strip `#`-comments and `\xa0` non-breaking-space (sample.docx has this — caused parser error, fix is `text.replace('\xa0', ' ')`), `ast.literal_eval` the dict.
3. **Per-source normalization** (different rules per docx):
   - sample: bbox already normalized; obstacle `info` is inconsistent (`[0.3, 13]` vs `[12, 0.7]`). Heuristic: whichever value is integer in `[1,12]` is the clock, the other is the distance.
   - 4471_data: bbox already normalized; obstacle `info` is `[distance, clock]` → swap.
   - comp4471: bbox in pixel coords at 4284×5712 → divide; obstacle `info` already `[clock, distance]`; obstacles HERE have bbox (unique to this source).
4. **Drop None entries**: `{"class": null, "bbox": null}` rows are skipped.
5. **Output unified JSON** at `data/HK_custom_for_finetuning/labels_unified.json`:
   ```json
   {
     "<filename>": {
       "source": "comp4471|4471_data|sample",
       "img_w": int, "img_h": int,
       "objects":  [{"class": str, "bbox": [x1,y1,x2,y2], "info": [h, clock, dist]}, ...],
       "obstacles":[{"class": "obstacle", "bbox": [x1,y1,x2,y2]|null, "info": [clock, dist]}, ...]
     }
   }
   ```

### 2.3 YOLO format conversion — `scripts/convert_to_yolo.py` (NEW)
Reads `labels_unified.json`, writes Ultralytics-format labels.

Standard YOLO label line (5 columns): `cls cx cy w h`
**Our extended label line (6 columns): `cls cx cy w h distance_m`**

The 6th column is needed by C (distance-weighted loss). Vanilla Ultralytics ignores extra columns, so the same files work for A; for C we use a custom dataloader that reads the 6th.

For **obstacles without bbox** (sample + 4471_data sources): we cannot train YOLO on these. Options:
- (i) Drop them entirely from training (lose ~5–8 supervisory signals).
- (ii) Use the obstacle's `info` direction + an estimated bbox heuristic (no — error-prone).
- **Decision: (i) drop.** Only obstacles with explicit bbox (comp4471 source) feed the trainer. The lost obstacles still appear in evaluation as "ground-truth distance/direction alerts we should produce" via the runtime obstacle_proposer.

Train/val split:
- Stratified by source so each split has comp4471 + 4471_data + sample mixed.
- ~22 train / ~6 val.
- Seed fixed for reproducibility.

Output:
```
data/HK_custom_for_finetuning/
  images/
    train/*.{jpg,png}
    val/*.{jpg,png}
  labels/
    train/*.txt    # 6-column format
    val/*.txt
  data.yaml        # 8-class names, paths
```

---

## 3. Training A — RGB-only fine-tune (baseline)

### 3.1 Reuses
`scripts/train_hazard_yolo.py` already exists. Updates needed:
- Class list: from current 6 hazard classes → new 8-class taxonomy (§1).
- Default `--data` path → `data/HK_custom_for_finetuning/data.yaml`.
- Otherwise unchanged: Ultralytics handles mosaic/flip/hsv/scale automatically.

### 3.2 Hyperparameters (initial)
| param | value | rationale |
|---|---|---|
| backbone | YOLOv8m | balance accuracy/speed; matches prior config |
| imgsz | 640 | ultralytics default; 4284×5712 photos downsample |
| epochs | 100 | ~28 imgs × 100 ≈ 2800 effective gradient steps with batch=16 |
| batch | 8 | conservative for laptop GPU/MPS |
| lr0 | 0.001 | small LR for transfer learning |
| patience | 30 | early stop |
| augmentation | mosaic=1.0, mixup=0.1, hsv_v=0.4, fliplr=0.5, scale=0.5 | aggressive given small dataset |
| close_mosaic | 10 | last 10 epochs without mosaic, lets model see clean compositions |

### 3.3 Output
- `runs/detect/hazard_yolov8m_A/weights/best.pt`
- mAP@0.5, mAP@0.5:0.95 in `results.csv` from Ultralytics

---

## 4. Training C — Distance-weighted loss

### 4.1 Mathematical form

For each ground-truth target with distance `d` (meters), multiply its per-target loss contribution by

```
w(d) = exp(-d / τ)        with τ = 2.0
```

| d (m) | w(d) |
|---|---|
| 0.5  | 0.78 |
| 1.0  | 0.61 |
| 2.0  | 0.37 |
| 5.0  | 0.082 |

Closer objects get up to ~10× more weight than far ones. Smooth (no kinks), bounded, doesn't blow up at d→0.

Targets without distance (e.g. obstacles whose `info[1]` is missing — none in current data, but defensive): w = 1.0 (no reweighting).

### 4.2 Where to apply

Ultralytics' `v8DetectionLoss` returns three components per target:
- `loss_iou` (CIoU box loss)
- `loss_cls` (BCE cls loss)
- `loss_dfl` (Distribution Focal Loss for box regression)

We multiply ALL THREE per-target by `w(d)`. Rationale: we want the network to predict closer objects' boxes, classes, and box-distribution accurately — not just one of those.

### 4.3 Implementation strategy

Instead of forking `ultralytics`, we subclass and inject:

```
src/depth_yolo/
├── __init__.py
├── dataset.py        # custom YOLODataset that reads 6th label column → t.distances
└── loss.py           # subclass v8DetectionLoss; overrides __call__ to multiply
                      # per-target loss by w(distances) before reduction
```

The training script picks up the custom loss via `model.loss = WeightedV8DetectionLoss(...)` after `YOLO(weights)` instantiates the model. (Ultralytics exposes `model.model.criterion`.) Exact attachment point will be verified in implementation Phase A by reading installed ultralytics version.

### 4.4 Risks
- Ultralytics internal API changes between versions. Pin `ultralytics==8.x.y` (currently installed) and document.
- DFL loss is computed per-anchor not per-target → need to map anchors back to assigned targets. Ultralytics does this in `BboxLoss`, we just multiply the assignment-weighted output.
- Validate by training 1 epoch with τ→∞ (i.e., w≡1): mAP should match A exactly. If not, the hook is broken.

---

## 5. Training B — RGB-D 4-channel input

### 5.1 Pre-compute depth maps once

`scripts/precompute_depth.py` (NEW):
- Loads `DepthEstimator` (metric-indoor)
- For each image in `images/{train,val}/`, save `depth/{train,val}/{stem}.npy` (float32 meters)
- Runs once; cached for all subsequent RGB-D training runs

### 5.2 Modify YOLOv8 first conv

YOLOv8 first conv has `in_channels=3`. We change to 4:

```
old_conv = model.model[0].conv
new_conv = nn.Conv2d(4, old_conv.out_channels,
                     kernel_size=old_conv.kernel_size,
                     stride=old_conv.stride,
                     padding=old_conv.padding,
                     bias=False)
# Init: copy RGB weights as-is, init 4th channel as MEAN of RGB weights
with torch.no_grad():
    new_conv.weight[:, :3] = old_conv.weight
    new_conv.weight[:, 3:4] = old_conv.weight.mean(dim=1, keepdim=True)
model.model[0].conv = new_conv
```

Rationale for mean-init: equivalent to "depth contributes a generic edge-like response from start" — preserves pretrained features, lets gradient adapt the depth channel during fine-tune.

### 5.3 Custom dataset

`src/depth_yolo/rgbd_dataset.py`:
- Subclass Ultralytics `YOLODataset.load_image()`.
- For each image, also load matching `.npy` depth map, resize/crop to match image augmentation pipeline, normalize to roughly the same scale as RGB pixels (depth in meters / 10 → ~[0, 1]).
- `__getitem__` returns `image[4, H, W]` instead of `[3, H, W]`.
- Augmentations that geometrically transform the image (flip, scale, mosaic) MUST be applied identically to the depth channel. Photometric augmentations (HSV) only on RGB channels.

### 5.4 Risks (real, not boilerplate)
- **Depth–RGB augmentation alignment** is the most likely place to break. Mosaic in particular composes 4 images at random positions — the depth maps must follow the exact same mosaic placement. We will test this with a debug visualizer before launching long training.
- **Depth normalization scale**: if we leave depth in [0, 10] meters, it dominates RGB which is in [0, 1]. Need to normalize depth to [0, 1] — divide by `10.0` (clamping max 10 m) is the simplest choice.
- **Pretrained features may be wasted**: depth is a fundamentally different signal from RGB. Mean-init might not help much. Fallback: zero-init the 4th channel and let it learn from scratch — slower but potentially better.

### 5.5 Decision — fail-safe ordering

We implement A → A+C → A+B+C in that order. If B fails after, say, 3 attempts, the report still has A and A+C as solid contributions and we mark B as future work. We do NOT block on B.

---

## 6. Inference pipeline (depth-first redesign)

### 6.1 Current flow
```
Image → YOLO → MobileSAM mask → Depth measure → Distance rank → Top-K alerts
```
Limitations:
- YOLO misses non-COCO classes (walls, etc.). Even after fine-tune, things outside our 8 classes are invisible.
- Result: user can collide with stuff the system never reported.

### 6.2 Proposed flow
```
Image
  ├── YOLO (fine-tuned 8-class)         ── parallel ──┐
  ├── Depth Anything V2 Metric-Indoor   ── parallel ──┤
  └── MobileSAM masks (per detection)   ────────────  │
                                                       ▼
                                              ┌────────────────┐
                                              │ obstacle_proposer │
                                              │ (depth-only)      │
                                              └────────────────┘
                                                       │
                              ┌─── classified detections (from YOLO, with distance)
                              │
                              ├─── unclassified close regions (from proposer)
                              ▼
                       merge + rank by distance
                              ▼
                       top 3–5 alerts
```

### 6.3 `src/obstacle_proposer.py` (NEW) — algorithm

Input:
- depth map (H, W) in meters
- list of YOLO-detected masks (binary H×W each)

Steps:
1. Build `unclaimed = depth_map.copy()`. Set `unclaimed[mask] = inf` for every YOLO mask (these regions are already accounted for).
2. Threshold: `close_mask = unclaimed < threshold_m` (default 2.0 m, configurable).
3. Connected-components on `close_mask`. Drop components below min area (e.g., 0.5% of image area — filters noise).
4. For each remaining component:
   - Compute centroid → direction (left/center/right) and clock angle via existing FOV math.
   - Compute representative distance: top_k_100 of depth values in component.
   - Emit as `{"class_name": "obstacle", "bbox": auto-from-component, "mask": component, "depth_stats": {...}, "direction": ..., "angle_deg": ...}`.

Output: list of detection-shaped dicts that can be merged into `analyze_scene()` results without changing downstream code.

### 6.4 `src/scene_analyzer.py` integration

Add an optional argument `obstacle_proposer: bool = False`. When `True`:
1. Run analyze_scene as today.
2. Compute obstacle proposals from the same depth map + collected masks.
3. Append proposals to results.

`run_scene_analysis.py` adds a `--include-obstacles` flag.

---

## 7. Evaluation

### 7.1 Held-out set

Same val split (~6 images) used during training serves as held-out for detection metrics. To avoid hyperparameter overfit we additionally hold out **2** images entirely (not seen by any training run, named `_test_only_`) for the final report numbers.

### 7.2 Metrics

| Metric | Computed by | What it captures |
|---|---|---|
| mAP@0.5 (per class) | Ultralytics built-in | detection accuracy |
| mAP@0.5:0.95 | Ultralytics built-in | detection accuracy strict |
| **Distance MAE (m)** | NEW eval script | predicted distance vs GT `info[2]` per matched detection |
| **Top-K ranking accuracy** | NEW eval script | overlap of system's top-3 nearest with GT-sorted top-3 nearest |
| **Obstacle recall** | NEW eval script | for env obstacles: was a detection or proposal raised within angle/distance tolerance |
| **NYU depth metrics** | already implemented | pure depth backbone validity (independent of fine-tune) |

`scripts/evaluate_hazard.py` (NEW): runs models against val (+ test) sets, emits a results table and per-image JSON.

### 7.3 Comparison table (final)

|             | mAP@0.5 | mAP@0.5:0.95 | Distance MAE | Top-3 ranking acc | Obstacle recall |
|-------------|---------|--------------|--------------|-------------------|-----------------|
| vanilla YOLOv8m   |         |              |              |                   |                 |
| A (RGB fine-tune) |         |              |              |                   |                 |
| A+C (RGB + dist-w loss) |   |              |              |                   |                 |
| A+B+C (RGB-D + dist-w loss) |    |          |              |                   |                 |

Goal: A+C beats A on Distance MAE / Top-K ranking; A+B+C beats A+C on at least one column. If neither: write up as negative result + analysis.

### 7.4 Failure mode catalog

For the report's Experiments section, collect:
- ≤5 visualized failure cases per model
- Categorized as: missed detection / wrong class / wrong distance / wrong direction
- Helps reviewer trust the numbers

---

## 8. File / module map (final)

```
project/
├── data/HK_custom_for_finetuning/
│   ├── raw_images/              # extracted from docx (NEW, gitignored)
│   ├── labels_unified.json      # parsed + normalized GT (NEW)
│   ├── images/{train,val}/      # YOLO layout (NEW)
│   ├── labels/{train,val}/      # 6-column YOLO labels (NEW)
│   ├── depth/{train,val}/       # precomputed metric depth .npy (NEW, for B)
│   └── data.yaml                # ultralytics config (NEW)
│
├── scripts/
│   ├── build_hk_dataset.py      # NEW — docx → labels_unified.json + raw_images/
│   ├── convert_to_yolo.py       # NEW — labels_unified.json → YOLO layout + train/val split
│   ├── precompute_depth.py      # NEW — RGB → depth .npy (for B)
│   ├── train_hazard_yolo.py     # MODIFY — supports --rgbd / --distance-weighted flags
│   └── evaluate_hazard.py       # NEW — comparison metrics
│
├── src/
│   ├── obstacle_proposer.py     # NEW — depth-only fallback proposer
│   ├── scene_analyzer.py        # MODIFY — add include_obstacles arg
│   ├── proximity_alerter.py     # unchanged
│   ├── hazard_scorer.py         # unchanged
│   └── depth_yolo/              # NEW — RGB-D + weighted-loss code
│       ├── __init__.py
│       ├── dataset.py           # custom YOLODataset reading 6th column + RGB-D stack
│       ├── model.py             # 4-channel first-conv surgery
│       └── loss.py              # WeightedV8DetectionLoss
│
└── docs/
    └── PLAN_DEPTH_AWARE_YOLO.md  # this document
```

---

## 9. Execution order (dependencies, not dates)

```
[1] build_hk_dataset.py    →  labels_unified.json + raw_images/
[2] convert_to_yolo.py     →  YOLO-format dataset
[3] obstacle_proposer.py   →  pipeline integration (independent of training)
[4] precompute_depth.py    →  depth .npy   (only needed for B)
[5] Train A
[6] Train A+C  (needs §4 loss module)
[7] Train A+B+C  (needs §5 model surgery + RGB-D dataset)
[8] evaluate_hazard.py against all 4 models
[9] write up results
```

[3] can run in parallel with anything from [4] onward. [5]/[6]/[7] are sequential because each builds on the previous.

---

## 10. Risks & mitigations

| ID | Risk | Likelihood | Mitigation |
|---|---|---|---|
| R1 | 28 images is too few; mAP variance huge | high | Heavy augmentation; pull public data if A's val mAP@0.5 < 0.4 |
| R2 | Ultralytics loss-hooking API changes break C | medium | Pin version; sanity-check by training 1 epoch with τ=∞ → must match A |
| R3 | `table` vs `dining table` confused (only ~3 imgs each) | medium | Track per-class P/R; merge if F1 below 0.3 |
| R4 | RGB-D mosaic alignment bug (B) | high | Build a debug visualizer that overlays depth on the mosaic image, eyeball BEFORE long training |
| R5 | Precomputed depth becomes stale if model changes | low | depth/{train,val}/ regenerated from a single script; document |
| R6 | Obstacle proposer over-segments → spammy alerts | medium | Min area + distance threshold tuned on val set |
| R7 | distance values in `info[2]` are estimates, not measurements | high (already true) | Acknowledge in report; absolute distance MAE is upper bound — report it as "agreement with annotator estimates" |

---

## 11. What this plan deliberately does NOT include

- **Real-time / mobile deployment** — out of scope, future work.
- **TTS / audio output** — out of scope.
- **Automatic pose estimation** ("which way is the person facing") — out of scope.
- **Multi-camera / video** — single image only.
- **Self-supervised pretraining on extra HK data** — out of scope.
- **Model export (ONNX / CoreML)** — out of scope unless time permits.

---

## 12. Approval checklist (review before implementation begins)

Please confirm or push back on each:

- [ ] **§1.1** keep `table` and `dining table` separate at start, merge only if F1 < 0.3 — OK?
- [ ] **§2.3** drop obstacles without bbox from training (keep them in evaluation) — OK?
- [ ] **§4.1** loss weight `w(d) = exp(-d/2)` — OK with τ=2 m?
- [ ] **§4.2** apply weight to ALL three loss components (iou + cls + dfl) — OK, or only iou?
- [ ] **§5.2** mean-init the 4th channel from RGB weights — OK, or zero-init?
- [ ] **§5.3** normalize depth by /10.0 to keep it in ~[0, 1] alongside RGB — OK?
- [ ] **§6.3** obstacle proposer threshold 2.0 m + min component 0.5% area — OK as starting point?
- [ ] **§7.1** hold out 2 images entirely from training (not just val) — OK with reducing training set to 26?
- [ ] **§9** execution order — anything to reorder?

Once these are checked, we begin Phase 1 ([1] → [2]).
