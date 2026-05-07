# Public-Data Integration Plan

**Status:** draft for review (do not start implementation until approved)
**Goal:** Lift YOLOv8m fine-tune mAP@0.5 from ~0.13 (28-image only) to a defensible
number on our HK indoor val/test set, by augmenting the training pool with
public Roboflow Universe / Open Images data — without compromising the
"evaluated on HK indoor" narrative for the report.

---

## 0. Why this is needed

After the A-baseline run we have:

| Metric | Value |
|---|---|
| mAP@0.5  | 0.1275 |
| mAP@0.5:0.95 | 0.0722 |
| Distance MAE | NaN (no preds at conf=0.10) |
| Top-3 ranking acc | NaN |
| Obstacle recall | 0 / 3 |

This is consistent with 21 training images on a 25M-parameter detector.
Best-case fix: more data. The detector is the bottleneck — depth pipeline
is fine and obstacle proposer is a separate fallback that doesn't depend
on the detector.

---

## 1. Hard rules (non-negotiable)

These rules protect the integrity of the experiment + the report's narrative.

### 1.1 HK split stays sacred — public data goes to **train only**

Current splits are fixed:
- `images/train` — 21 HK images   ← we will INJECT public data here
- `images/val`   —  5 HK images   ← stays HK-only
- `images/test`  —  2 HK images   ← stays HK-only

**Why:** the report claims "evaluated on novel HK indoor data". If public
images leak into val/test that claim is dead. mAP also becomes uninformative
(public data is more uniform / easier than self-captured HK photos).

### 1.2 Public-data rows train detection only

Public images do not have:
- `info[2]` distance (no metric ground truth)
- `info[1]` clock direction (no ground truth)

So when we feed them into the trainer, their per-target distances are `NaN`.
Our `distance_weight()` already maps `NaN → 1.0` (no reweighting), so the C
variant works gracefully on mixed data — but the public examples don't
contribute distance signal, only detection signal.

### 1.3 License audit before pulling

Only datasets with explicitly permissive licenses go in:
- CC-BY (most Roboflow Universe content)
- CC0 / public domain
- MIT / Apache
- Open Images V7 (Apache 2.0)

We capture a `licenses.txt` listing every external dataset URL + license, to
cite in the final report.

### 1.4 Class mapping is the source of truth

A single config file `data/HK_custom_for_finetuning/class_mapping.yaml`
lists, per source dataset, how each public class maps to our 8 classes
(or `null` to drop). Reviewers can audit it; mistakes are fixable in one
file without re-downloading.

### 1.5 No "obstacle" augmentation from public data

Our `obstacle` class is a deliberately HK-specific catch-all (walls, drying
racks, wardrobes, etc.). Public obstacle datasets have a different
distribution — typically traffic cones / construction, not indoor stuff.
Mapping public "obstacle" to our `obstacle` would teach the wrong concept.

So `obstacle` stays at 11 HK annotations. The other 7 classes
(chair, table, refrigerator, door, bed, couch, dining_table) absorb public
data.

---

## 2. Dataset selection

### 2.1 Selection criteria

For each candidate Roboflow / Open Images dataset, evaluate on:

| Criterion | Weight | Notes |
|---|---|---|
| Class overlap | high | At least one of our 7 augmentable classes must be present |
| Domain | high | 1st-person indoor preferred. 3rd-person furniture catalogs OK as auxiliary. AVOID outdoor, cartoon, product-on-white |
| Image count | medium | 200–2000 per dataset; we want diversity, not volume |
| License | gate | Must be permissive |
| Annotation quality | medium | Spot-check 10 random images before committing |

### 2.2 Candidate datasets (search terms — verify on Roboflow Universe)

User runs these searches at https://universe.roboflow.com :

| Search | Likely yield | Target classes |
|---|---|---|
| `indoor furniture detection`     | several | chair, table, couch, bed |
| `home appliance detection`        | several | refrigerator |
| `door detection`                  | several | door |
| `chair detection indoor`          | many | chair |
| `office room objects`             | several | chair, table |

Open Images V7 (https://storage.googleapis.com/openimages/web/index.html)
has labels for all 7 of our augmentable classes — but it's millions of
images. We DO NOT pull all of it. We sample ~200–500 images per class
matching `indoor` filters via the FiftyOne CLI.

### 2.3 Decision rule

Pick **2–4 datasets** total adding **300–1500 images**. No more.

Why cap: 3000+ public images on top of 21 HK images → public data dominates,
val/test domain shift becomes a problem, training time grows. The point is
to give the model enough examples per class to learn `chair` reliably,
not to win Pascal VOC.

---

## 3. Class mapping

### 3.1 Schema — `data/HK_custom_for_finetuning/class_mapping.yaml`

```yaml
# Maps {external_dataset_name: {external_class_name: our_class_name_or_null}}
#   our_class_name must be one of the 8 in convert_to_yolo.HAZARD_CLASSES, or null to drop.
# Comments explain WHY each mapping was chosen (or rejected).

datasets:
  roboflow_indoor_furniture_v3:
    sourceURL: "https://universe.roboflow.com/<...>"
    license: "CC-BY 4.0"
    notes: "300 images, 1st-person indoor"
    classes:
      chair:        chair
      armchair:     chair       # flatten — our taxonomy doesn't separate
      stool:        chair
      sofa:         couch       # naming variant
      coffee_table: table
      dining_table: dining_table
      tv_stand:     null        # not in our taxonomy
      bed:          bed
      lamp:         null
      window:       null

  open_images_v7_indoor:
    sourceURL: "https://storage.googleapis.com/openimages/..."
    license: "Apache 2.0"
    notes: "FiftyOne sample, indoor filter, 500 imgs"
    classes:
      Chair:           chair
      Table:           table
      Couch:           couch
      Refrigerator:    refrigerator
      Bed:             bed
      Door:            door
```

### 3.2 Edge cases

- **Missing class in mapping** → log warning, drop that bbox (don't crash)
- **`null` mapping** → bbox dropped silently
- **Duplicate filename across datasets** → prefix by dataset name
  (`<dataset>__<original_filename>`)

---

## 4. Merge mechanics

### 4.1 Inputs (provided by user)

User downloads each dataset from Roboflow as **YOLOv8 export** which gives:
```
<download_dir>/
  data.yaml                # has nc, names
  train/images/*.jpg
  train/labels/*.txt       # standard 5-col cls cx cy w h
  valid/images/*.jpg       # we IGNORE these — never go into our val
  valid/labels/*.txt
  test/images/*.jpg        # we IGNORE
  test/labels/*.txt
```

User runs:
```bash
python3 scripts/merge_external_dataset.py \
    --source ~/Downloads/indoor_furniture_v3 \
    --dataset-name roboflow_indoor_furniture_v3 \
    --mapping data/HK_custom_for_finetuning/class_mapping.yaml
```

### 4.2 What the script does

1. Read `<source>/data.yaml` to learn the external `names` (id → name).
2. Look up the dataset's mapping in `class_mapping.yaml`. If missing, fail loud.
3. For every (`<source>/train/images/<f>`, `<source>/train/labels/<f>.txt`):
   a. Read the 5-col label file.
   b. For each row, look up `external_id → external_name → our_class_name`.
      - If `null` or unmapped: drop the row.
   c. Skip the image if 0 rows survive.
   d. Copy image to `images/train/<dataset>__<f>` (prefixed for unique filenames).
   e. Write remapped label to `labels/train/<dataset>__<stem>.txt`.
   f. Append `<dataset>__<stem>: [NaN, NaN, ...]` (one NaN per kept row) to a
      buffer that gets merged into `distances.json` at the end.
4. Append a row to `licenses.txt` with the dataset name + URL + license.
5. Print summary: `<dataset>: ingested N images, M labels, dropped K classes`.

The script only modifies the **train** split. `val/`, `test/`, and
`distances.json` for `val`/`test` keys are never touched.

### 4.3 Idempotency

Running twice with the same dataset overwrites the prefix-namespaced files
identically. We do NOT delete pre-existing public files when re-running,
which means changing the mapping requires manual cleanup. We document this
in the script's `--help`.

`--clean-existing` flag: removes all files matching `<dataset>__*` before
ingest, for clean re-runs.

---

## 5. Distance handling

### 5.1 distances.json after merge

```json
{
  "train": {
    "4471_data__16":                          [1.0, 1.2, 1.5],
    "comp4471__01":                           [1.27, 1.23],
    "roboflow_indoor_furniture_v3__abc123":   [NaN, NaN],
    ...
  },
  "val":  { "4471_data__09": [...], ... },
  "test": { ... }
}
```

JSON doesn't natively serialize `NaN`, so we use `null` and the loader maps
`null → NaN` during read. Update needed in
`src/depth_yolo/dataset.load_distances_for_split` — currently it returns
the raw list. Add a normalization step.

### 5.2 Loss behavior on mixed batches

`distance_weight(NaN, tau) → 1.0` (already implemented in `src/depth_yolo/loss.py`).
A batch containing 4 HK targets and 8 public targets:
- Compute `w_i = exp(-d_i / τ)` for HK, `w_i = 1.0` for public
- `scalar = mean(w_i)` is what `WeightedV8DetectionLoss.__call__` already does

Net effect: distance-weighted loss reweights the HK contribution but lets
public data contribute at full strength (which is what we want — public
data is mostly there to give the detector enough box-regression signal).

---

## 6. Re-training and re-evaluation

### 6.1 Train command (no change)

```bash
# Variant A on combined data
python3 scripts/train_hazard_yolo.py --epochs 100

# Variant A+C
python3 scripts/train_hazard_yolo.py --epochs 100 --distance-weighted
```

The trainer doesn't know or care about the source of training images.

### 6.2 Hyperparameter notes

With ~500–1500 training images instead of 21:
- Increase `--patience` to 50 (was 30) — more data → more room to improve
- Maybe lower lr0 slightly, but 0.001 is a fine starting point
- Mosaic / augmentation defaults stay on

### 6.3 Eval command (no change)

```bash
python3 scripts/evaluate_hazard.py \
    --weights runs/detect/<latest>/weights/best.pt \
    --conf 0.10 --tag A_combined --output results/metrics/eval_A_combined.json
```

### 6.4 Expected outcomes

Realistic targets after public-data merge (HK val set, 5 images, 9 instances):

| Metric | Before (HK only) | After (combined) realistic |
|---|---|---|
| mAP@0.5 | 0.13 | 0.40 – 0.65 |
| mAP@0.5:0.95 | 0.07 | 0.25 – 0.45 |
| Distance MAE | NaN | 0.5 – 1.5 m |
| Top-3 ranking | NaN | 0.40 – 0.70 |
| Obstacle recall | 0/3 | unchanged-ish (we didn't augment obstacle) |

These ranges are based on transfer-learning-on-mid-size-data norms. If
combined mAP@0.5 < 0.30, something is wrong with the merge (class mapping bug,
domain gap too large, etc.) — debug before reporting.

---

## 7. Risks

| ID | Risk | Mitigation |
|---|---|---|
| RP1 | Class definition drift (their "chair" includes outdoor benches) | Spot-check 10 random labels per dataset before committing the mapping |
| RP2 | Public images dominate, model overfits to product-photo aesthetic | Cap total at ~1500; sample if needed; prefer 1st-person indoor sources |
| RP3 | Domain gap → mAP on HK val drops vs. HK-only | If this happens, weight HK examples up via Ultralytics' `class_weights` or oversample HK in dataloader |
| RP4 | License oversight → reviewer catches it | Track every URL+license in `licenses.txt`, cite in report |
| RP5 | Filename collision between datasets | Prefix with `<dataset>__` always |
| RP6 | Roboflow YAML schema drift | Parse defensively; fall back to numbered classes if names absent |

---

## 8. Module layout

New / modified files:

```
project/
├── data/HK_custom_for_finetuning/
│   ├── class_mapping.yaml             # NEW — maps each external dataset's classes to ours
│   ├── licenses.txt                    # NEW — append-only ledger of external sources
│   └── (everything else unchanged)
│
├── scripts/
│   └── merge_external_dataset.py      # NEW — the core merger
│
├── src/depth_yolo/
│   └── dataset.py                     # MODIFY — load_distances_for_split must
│                                      #          coerce JSON null → NaN
│
├── tests/
│   └── test_merge_external.py         # NEW — unit tests for class mapping +
│                                      #       end-to-end merge into a tmp dataset
│
└── docs/
    └── PLAN_PUBLIC_DATA_MERGE.md      # this document
```

---

## 9. Execution order (dependency-driven)

```
[1] Update load_distances_for_split to handle null → NaN
[2] Write merge_external_dataset.py (with --dry-run mode)
[3] Write class_mapping.yaml skeleton + 1 example dataset entry
[4] Write tests/test_merge_external.py + run
[5] User: search Roboflow Universe, pick 2–4 datasets, fill class_mapping.yaml
[6] User: download → run merge script → audit output
[7] User: retrain → evaluate
[8] If needed, iterate on class_mapping.yaml + re-merge
```

Steps [1]–[4] are mine and unblocking; I can finish them now without
waiting on the user's Roboflow choices. Steps [5]–[8] are the user's loop.

---

## 10. What this plan deliberately does NOT include

- Auto-downloading via Roboflow API — needs user-specific API keys; better
  to keep the human in the loop on dataset selection.
- Image quality filtering (deduplication, blur detection, etc.) — over-engineering
  for our scale.
- Full Open Images V7 ingestion — too large; user can sample via FiftyOne
  separately and run the merge script on the sampled folder.
- Re-running variants C and B until A on combined data is stable.

---

## 11. Approval checklist

Confirm or push back on each:

- [ ] **§1.1** public data injected into TRAIN only; val/test stays HK-only — OK?
- [ ] **§1.5** do NOT augment `obstacle` from public data — OK?
- [ ] **§2.3** cap total external additions at 300–1500 images — OK number?
- [ ] **§3.1** class mapping in a single yaml, not in code — OK?
- [ ] **§5.1** use JSON `null` for missing distance, loader coerces to NaN — OK?
- [ ] **§9** I do steps [1]–[4] now, you do [5]–[8] when ready — OK?

Once these are checked, I implement [1]–[4]. You then do the Roboflow legwork.
