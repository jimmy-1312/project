# Final experiment plan — HazardSight

**Status:** locked-in design (5 decisions confirmed)
**Goal:** evaluate whether a generic-trained indoor detector + frozen depth backbone
generalizes to HK 1st-person photos, with optional movement-guidance layer.

---

## 0. Locked-in decisions

| # | Decision | Choice |
|---|---|---|
| 1 | NYU role | **depth backbone validation only** (no YOLO training on NYU) |
| 1 | YOLO training data | Roboflow `indoor_furniture_v3` (~733 images, our 8 classes via mapping) |
| 1 | Validation strategy | **10-fold cross-validation on Roboflow** |
| 1 | Test set | **HK 28 photos** — never seen during training (held-out domain test) |
| 2 | Fine-tuning method | **Detection head only** (`freeze=10`, backbone frozen) |
| 3 | Distance weighting | **c4 only** — post-hoc at inference (already in `proximity_alerter`); no training-time modification |
| 4 | Step mode | **Movement guidance**: given detected objects' positions/distances, suggest where the user should move (forward / left / right / stop). Inference-time only. |
| 5 | Backbone | **YOLOv11s** (9.4M params) for everything |
| 5 | Test on | All 28 HK photos (no train leakage; HK is held-out test domain) |

---

## 1. Data restructure

### 1.1 Current state (after restratify)

```
images/train  609   HK 22 + Roboflow 587  ← HK leaked into train
images/val     76   HK  3 + Roboflow  73  ← HK leaked into val
images/test    76   HK  3 + Roboflow  73  ← HK leaked into test
```

This breaks decision §0.5 ("HK never seen during training"). Must restructure.

### 1.2 New split (target)

```
images/test         28   HK only        ← held-out domain test (never used in training)
images/roboflow    733   Roboflow only  ← split into 10 folds at training time

(no fixed train/val — k-fold creates them dynamically)
```

### 1.3 New script — `scripts/prepare_kfold.py`

Does the restructure in one go:
1. Find all images, classify by source (HK prefix vs `indoor_furniture_v3__` prefix).
2. Move all HK → `images/test`, all labels → `labels/test`.
3. Move all Roboflow → `images/roboflow`, labels → `labels/roboflow`.
4. Regenerate `distances.json` keyed correctly (HK in `test`, Roboflow has NaN distances).
5. Write `kfold_splits.json` — for each fold k=0..9, a list of train/val image stems.
6. Update `data.yaml` to point at the new structure with `test: images/test`.

Stratification within Roboflow: not needed (single source). Just shuffle with seed and slice.

---

## 2. Stage-by-stage plan

### Stage 1 — Vanilla baseline on HK (no training)

**Goal:** establish lower bound. What does pretrained YOLOv11s (COCO 80 classes) get on our HK data without any fine-tune?

**Why mAP will be misleading:**
- Pretrained YOLO knows COCO's 80 classes.
- Our HK GT has 8 classes; only some overlap (chair, dining_table, refrigerator, bed, couch).
- Classes with NO COCO equivalent (`door`, `obstacle`) → recall ≡ 0 by construction.

**Class mapping for fair eval (HK GT id → COCO predict id):**

| HK class (id) | COCO id | COCO name | Note |
|---|---|---|---|
| chair (0) | 56 | chair | direct |
| table (1) | 60 | dining table | best match |
| refrigerator (2) | 72 | refrigerator | direct |
| door (3) | — | — | no COCO equivalent → recall=0 |
| bed (4) | 59 | bed | direct |
| couch (5) | 57 | couch | direct |
| dining_table (6) | 60 | dining table | direct |
| obstacle (7) | — | — | no COCO equivalent → recall=0 |

**Reporting metrics:**
- **Per-class precision, recall, F1** at IoU=0.3 (loose match, 1st-person photos)
- mAP@0.5 secondary (will be capped by missing COCO classes)
- Class-by-class P/R/F1 table — explicitly shows which classes the baseline can/can't handle

**New script:** `scripts/eval_vanilla_baseline.py` — runs vanilla YOLOv11s on HK test, applies COCO→HK class mapping for matched classes only, reports per-class P/R/F1.

**Time:** ~10 min (no training, just inference + matching).

### Stage 2 — Detection-head fine-tuning on Roboflow (10-fold CV)

**Goal:** generic indoor detector for our 8 classes, with rigorous validation.

**Approach:**

```
For k in 0..9:
    train_imgs, val_imgs = kfold_splits[k]
    write tmp_data.yaml pointing at images/roboflow with this split
    YOLO11s.train(
        data=tmp_data.yaml,
        freeze=10,              # ← KEY: backbone frozen, head only
        epochs=30,
        imgsz=640,
        batch=16,
        lr0=<sweep>,
        patience=15,
        name=f"hazard_kfold_{k}",
    )
    metrics[k] = evaluate on val_imgs
```

**Hyperparameter sweep (3 configs × 10 folds = 30 runs):**
- lr0 ∈ {1e-3, 5e-4, 1e-4}
- (other params fixed: epochs=30, batch=16, imgsz=640)

**Reporting:**
- For each lr config: mean ± std of val mAP@0.5 across 10 folds
- Pick best lr by mean
- Best lr's per-class mAP, P, R averaged over folds

**Final model selection:**
- **Take the BEST lr config**
- Train on ALL Roboflow (no held-out val) using best lr → final weights for HK test
- Or: ensemble of 10 fold weights — averaged predictions (more rigorous, slower at inference)

**Decision rule (per §0):**
- If mean fold mAP ≥ vanilla baseline mAP on Roboflow (we'll need vanilla on Roboflow too) → fine-tune wins, use it
- Else → revert to vanilla, document as "fine-tune did not exceed baseline"

**New scripts:**
- `scripts/kfold_train.py` — runs 10 folds with given lr
- `scripts/kfold_aggregate.py` — collects 10 fold results, computes mean/std, picks best
- Modify `train_hazard_yolo.py` — add `--freeze N` flag (delegates to ultralytics `freeze`)

**Time:** 10 folds × ~20 min × 3 configs = ~10 hours total, parallelizable per fold if multiple GPUs. On M4 single device: sequential. Can reduce to 5-fold × 2 configs = ~3 hours for first pass.

### Stage 3 — HK test (held-out, never trained on)

**Goal:** test domain transfer. Generic-trained model on Roboflow → applied to HK 1st-person photos.

**Approach:**

```
final_weights = best_kfold_model.pt
For each HK image (28 total):
    predict with conf=0.01 (capture all reasonable detections)
    extract bbox + class + depth from full pipeline
For evaluation:
    precision, recall, F1 per class at IoU=0.3
    Distance MAE on matched detections (use our HK distance GT)
    Top-K nearest recall (uses depth + GT distance)
```

**Reporting (the centerpiece table):**

| Metric | Vanilla YOLO11s (Stage 1) | Fine-tuned (Stage 2 best) |
|---|---|---|
| chair P / R / F1 | … | … |
| table P / R / F1 | … | … |
| refrigerator P / R / F1 | … | … |
| door P / R / F1 | 0 / 0 / 0 (no COCO) | … |
| bed P / R / F1 | … | … |
| couch P / R / F1 | … | … |
| dining_table P / R / F1 | … | … |
| obstacle P / R / F1 | 0 / 0 / 0 (no COCO) | … |
| Distance MAE (m) | … | … |
| Top-3 nearest recall | … | … |

This table tells the story: "fine-tuning recovers the 2 classes baseline can't see, and improves overall on HK domain."

**Time:** ~30 min.

### Stage 4 — NYU depth backbone validation

**Goal:** show our depth backbone (Depth Anything V2 Metric-Indoor) is SOTA-level, regardless of detection.

**Approach:** existing `scripts/evaluate_depth_nyu.py`, no changes needed.

```bash
pip3 install datasets
python3 scripts/evaluate_depth_nyu.py \
    --hf-dataset sayakpaul/nyu_depth_v2 --split validation --max-images 200 \
    --output results/metrics/depth_nyu.json
```

**Reporting metrics (standard depth-paper format):**
- AbsRel, RMSE, RMSE_log, δ<1.25, δ<1.25², δ<1.25³

These compare directly to Depth Anything V2 paper Table 2. Expected: AbsRel ≈ 0.06, δ1 ≈ 0.95.

**Time:** ~30 min (200 images on M4).

### Stage 5 — Movement guidance ("step mode" reframed)

**Goal:** post-process detections into a movement instruction. Inference-time only.

**Logic (rule-based, deterministic):**

```
Input: top-K nearest detections from proximity_alerter
       (each has: class, distance_m, direction in {left, center, right})

1. CRITICAL_DIST = 0.5 m   # anything closer → STOP
2. CAUTION_DIST  = 1.5 m   # anything closer → adjust path

closest = nearest[0]

if closest.distance_m < CRITICAL_DIST:
    return "STOP"

if closest.direction == "center" and closest.distance_m < CAUTION_DIST:
    # something dead ahead within 1.5 m
    # find clearer side
    left_blocked  = any(d.direction == "left"  and d.distance_m < CAUTION_DIST for d in nearest)
    right_blocked = any(d.direction == "right" and d.distance_m < CAUTION_DIST for d in nearest)
    if not left_blocked and right_blocked:
        return "MOVE LEFT"
    if not right_blocked and left_blocked:
        return "MOVE RIGHT"
    if not left_blocked and not right_blocked:
        return "MOVE LEFT (default)"   # tie-break
    return "STOP — both sides blocked"

if closest.direction == "left" and closest.distance_m < CAUTION_DIST:
    return "DRIFT RIGHT"
if closest.direction == "right" and closest.distance_m < CAUTION_DIST:
    return "DRIFT LEFT"

return "FORWARD CLEAR"
```

**Output examples:**
- 1 chair at 0.4 m center → "STOP"
- 1 sofa at 1.2 m center, no other → "MOVE LEFT"
- 1 chair at 1.0 m left, clear right → "DRIFT RIGHT"
- 1 chair at 3 m right → "FORWARD CLEAR"

**New file:** `src/navigation_guide.py`. Single function `suggest_movement(top_k_alerts) -> str`. Plug into `run_scene_analysis.py` after `proximity_alerter`, dump in JSON as `"movement"`.

**Demo for slides/report:** generate 4 example images covering the 4 instructions.

**Time:** ~1 hour (code + tests + demo).

---

## 3. Per-stage time budget (estimate)

| Stage | Wall-clock |
|---|---|
| 0. Data restructure (`prepare_kfold.py`) | 30 min code + 5 min run |
| 1. Vanilla baseline on HK | 10 min |
| 2. 10-fold CV (3 lr × 10 folds × ~20 min) | ~10 hours (or 5-fold × 2 lr = 3 hr first pass) |
| 3. Final test on HK | 30 min |
| 4. NYU depth eval | 30 min |
| 5. Movement guidance | 1 hour |
| **report / slide updates** | 1 day (separate) |

**Critical path:** Stage 2 dominates. Recommend 5-fold × 2 lr configs first; if results promising, expand to 10-fold × 3 lr for the report.

---

## 4. New / modified files (concrete list)

### NEW

```
scripts/prepare_kfold.py        — restructure dataset for k-fold + HK held-out
scripts/kfold_train.py          — run 10 folds, save per-fold weights+metrics
scripts/kfold_aggregate.py      — mean ± std across folds, pick best
scripts/eval_vanilla_baseline.py — vanilla YOLO with COCO→HK class mapping
src/navigation_guide.py         — movement guidance (rule-based)
docs/PLAN_FINAL.md              — this document
```

### MODIFIED

```
scripts/train_hazard_yolo.py    — add --freeze N flag (default 0, set to 10 for head-only)
scripts/run_scene_analysis.py   — call navigation_guide after proximity_alerter, add to JSON
src/proximity_alerter.py        — minor: expose distance_to_steps cleanly for navigation_guide
tests/                          — add test_navigation_guide.py (~10 simple cases)
```

### DELETE / OBSOLETE

```
scripts/oversample_hk.py        — already deleted
scripts/restratify_splits.py    — superseded by prepare_kfold.py (delete after migration)
```

---

## 5. Risks / things to watch

| ID | Risk | Mitigation |
|---|---|---|
| R1 | `freeze=10` value wrong for YOLO11s | Verify by inspecting model.named_parameters() before training; adjust if needed |
| R2 | k-fold disk overhead (10 copies of weights) | Save only best.pt per fold, delete intermediate |
| R3 | Vanilla baseline COCO mapping ambiguous | Document explicit mapping table; report per-class so no hidden averaging |
| R4 | Roboflow Universe URL deprecated | Already downloaded locally; cite both URL + local snapshot |
| R5 | Movement guidance over-fits to thresholds | Keep rule simple, document thresholds in `config.py`, expose as CLI flags |
| R6 | NYU sayakpaul/nyu_depth_v2 schema changes | Hardcode field names in evaluate_depth_nyu defensively (already done) |

---

## 6. Stop conditions / decision points

- **After Stage 1:** if vanilla mAP@0.3 on HK is already > 0.4 for a class, we know that class is "easy"; focus fine-tune on the hard classes (door, obstacle).
- **After Stage 2 first fold:** if val mAP < 0.10 → stop sweep, debug. Likely freeze count or lr issue.
- **After Stage 2 all folds:** if mean ± std overlaps vanilla → revert weights, report "no significant improvement from head-only fine-tune".
- **After Stage 3:** results table goes into report regardless of outcome.

---

## 7. Reporting

Once Stages 1–4 done, the report's Experiments section writes itself:

> §5.1 Depth backbone validation on NYU Depth v2 (Stage 4)  
> §5.2 Vanilla baseline on HK photos (Stage 1) — class mapping, per-class P/R/F1  
> §5.3 Detection-head fine-tuning with 10-fold CV (Stage 2) — mean ± std, best config  
> §5.4 Domain-transfer test on HK (Stage 3) — final P/R/F1 vs baseline + Distance MAE  
> §5.5 Movement guidance examples (Stage 5) — 4 qualitative cases  
> §5.6 Limitations and negative findings — small HK test, COCO mapping caveats

---

## 8. Approval checklist

- [ ] Class mapping in §1 (Stage 1 baseline) — OK?
- [ ] 10-fold CV vs 5-fold first pass — go straight to 10 or 5-first-then-10?
- [ ] Movement guidance thresholds (CRITICAL=0.5m, CAUTION=1.5m) — OK as defaults?
- [ ] After Stage 2, take "best fold weights" vs "retrain on all Roboflow with best lr" — preference?

Once these are confirmed, I start with Stage 0 (`prepare_kfold.py`).
