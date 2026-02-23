# Pipelines Deep Dive (SAM3, DINOv3, T‑LEAP → ML)

## Purpose of this document

This is the **step-by-step** explanation of how the system transforms a video into features and predictions. It’s written to match the **actual code paths** in this repo.

Key code references:
- NATS subjects + datastore config: `shared/config/config.yaml`
- SAM3: `services/sam3-pipeline/app/main.py`
- DINOv3: `services/dinov3-pipeline/app/main.py`
- T‑LEAP: `services/tleap-pipeline/app/main.py`
- ML: `services/ml-pipeline/app/main.py`
- Tracking/ID: `services/tracking-service/app/main.py`

---

## End-to-end flow (high level)

### Stage 0 — Data in

- A raw video is uploaded and becomes `{video_id}`.
- The system produces a **processed clip** (cropped/stabilized depending on config) that downstream pipelines read.
- For long videos, the **long-video-pipeline** segments and filters clips, then publishes good clips on `video.preprocessed`.

### Stage 1 — Per-video feature pipelines (parallel)

For each `video_id`, these pipelines run (mostly in parallel):

- **YOLO**: cow detections (bboxes)
- **SAM3**: segmentation mask features (shape)
- **DINOv3**: visual embeddings + similarity evidence
- **T‑LEAP**: keypoints + locomotion features

### Stage 2 — Identity (tracking / Re‑ID)

- **Tracking service** consumes YOLO + DINOv3 outputs and produces:
  - per-video tracks (within-video)
  - cross-video cow identity assignment (persistent cow ID)

### Stage 3 — Predictors

Downstream predictors consume aggregated features:

- **ML pipeline**: tabular features → CatBoost/XGBoost/LightGBM (+ ensemble)
- **TCN / Transformer**: time-series from pose features
- **GNN / Graph Transformer**: relational context via similarity graphs

### Stage 4 — Fusion

Fusion combines the upstream predictor outputs into a final probability and confidence.

---

## Where data is stored (practical mental model)

### Filesystem outputs (most pipelines)

Each pipeline writes a JSON result file:

- `data/results/<pipeline>/{video_id}_{pipeline}.json`

and publishes a NATS message including:

- `video_id`
- `pipeline`
- `results_path`
- `features` (aggregate features)

### Vector DB (Qdrant)

The DINOv3 pipeline stores vectors for similarity search / Re-ID:

- collection: `cow_embeddings` (default)
- point id: `video_id`
- vector: average embedding over sampled frames

### Relational DB (Postgres)

The tracking service persists identity:

- `cow_identities` (persistent cows)
- `track_history` (video sightings of cows)
- `lameness_records` (cow-level lameness history, when written)

---

## SAM3 pipeline (segmentation → shape features)

### Inputs

- Trigger subject: `video.preprocessed`
- Reads:
  - processed video file (`processed_path`)
  - YOLO results file for bboxes: `data/results/yolo/{video_id}_yolo.json`

### Core algorithm

1. **Frame sampling**: ~2 FPS
   - `frame_interval = max(1, fps // 2)`
2. **BBox prompt** (per sampled frame)
   - finds matching YOLO detection for that frame
3. **Segmentation**
   - If a checkpoint exists under `/app/shared/models/sam3/*.pth`, uses a SAM-style predictor with bbox prompt
   - Else, uses a **fallback mask** equal to the bbox rectangle
4. **Mask-to-features**
   - `mask_area`, `area_ratio`
   - contour-based `circularity = 4πA / P²`
   - `aspect_ratio` (contour bbox)
5. **Aggregate across frames**
   - averages the key shape features

### Output JSON

File: `data/results/sam3/{video_id}_sam3.json`

- `segmentations[]`: per processed frame
- `aggregated_features`:
  - `avg_mask_area`
  - `avg_area_ratio`
  - `avg_circularity`
  - `avg_aspect_ratio`

### Why it helps lameness

SAM3 features act like **shape priors** for gait:
- stance width / body profile changes can correlate with discomfort
- segmentation helps downstream models normalize and detect occlusions/partial views

---

## DINOv3 pipeline (embeddings → similarity evidence → Qdrant)

### Inputs

- Trigger subject: `video.preprocessed`
- Reads: processed video file (`processed_path`)
- Writes: `data/results/dinov3/{video_id}_dinov3.json`
- Stores vectors in Qdrant (`QDRANT_URL`, default from config)

### Core algorithm

1. **Frame sampling**: ~1 FPS
   - `frame_interval = max(1, fps)`
2. **Embedding per sampled frame**
   - preprocess via HuggingFace `AutoImageProcessor`
   - model via `AutoModel`
   - embedding = mean pooling of `last_hidden_state`
3. **Canonical frames**
   - selects first / middle / last sampled embeddings
4. **Average embedding**
   - `avg_embedding = mean(all_sampled_embeddings)`
5. **Similarity search**
   - query top-k in Qdrant
6. **Neighbor evidence**
   - if the retrieved neighbors have labels, compute fraction lame
   - otherwise fall back to 0.5 (“unknown”)
7. **Persist**
   - store `avg_embedding` to Qdrant under point id `video_id`

### Output JSON

File: `data/results/dinov3/{video_id}_dinov3.json`

- `embedding_dim`
- `num_embeddings`
- `canonical_frames[]` (each contains `frame`, `time`, `embedding`)
- `similar_cases[]` (from Qdrant search)
- `neighbor_evidence`

### Why it helps lameness

Embeddings provide a **visual similarity prior**:
- “this clip looks like previous clips labeled lame/healthy”
- also serves Re-ID / cow identity when combined with tracking

---

## T‑LEAP pipeline (pose → locomotion features)

### Inputs

- Trigger subject: `video.preprocessed`
- Reads:
  - processed video if available
  - otherwise falls back to raw video under `data/videos/{video_id}.*`

### Pose estimation strategy

The pipeline uses a **hybrid** approach:

- If `/app/data/models/cow_pose_roboflow.pt` exists:
  - runs YOLOv8 pose inference → 20 keypoints
  - uses heuristic keypoints when confidence is low (per keypoint threshold)
- Else:
  - uses YOLO detection only + full heuristic keypoint placement

### Locomotion features computed

Sampling: ~5 FPS (`frame_interval = max(1, int(fps // 5))`)

From the pose time series it computes:

- **Back arch**:
  - angle at withers using throat → withers → tailbase
  - `back_arch_mean`, `back_arch_std`, `back_arch_score`
- **Head bob**:
  - `head_bob_magnitude` = std of vertical head position
  - `head_bob_frequency` proxy
  - `head_bob_score` normalized
- **Stride (per leg)**:
  - mean/std of absolute x-deltas of hoof positions:
    - `stride_fl_mean/std`, `stride_fr_mean/std`, `stride_rl_mean/std`, `stride_rr_mean/std`
- **Asymmetry**:
  - `front_leg_asymmetry`, `rear_leg_asymmetry` as normalized left/right diffs
- **Overall lameness_score**
  - mean of available components (normalized)

### Output JSON

File: `data/results/tleap/{video_id}_tleap.json`

- `pose_sequences[]`: per processed frame (bbox + keypoints)
- `locomotion_features`: the computed features above
- `model_type`: `trained` vs `heuristic`

---

## Tracking service (ID / cow registry)

### Inputs

- `pipeline.yolo`: per-frame detections
- `pipeline.dinov3`: embeddings (and canonical frames)

### Outputs

- `data/results/tracking/{video_id}_tracking.json`
- Postgres:
  - creates/updates cow identity in `cow_identities`
  - creates sightings in `track_history`

### Conceptual algorithm

- **Within-video tracking**: associate bboxes across frames (ByteTrack-like)
- **Cross-video Re‑ID**:
  - compute/query embedding similarity in Qdrant
  - assign the current track to an existing cow or create a new cow

This enables per-cow aggregation and the Cow Registry UI.

---

## ML pipeline (feature vector → ensemble prediction)

### Inputs

The ML pipeline loads upstream JSON results from:

- `data/results/yolo/{video_id}_yolo.json`
- `data/results/sam3/{video_id}_sam3.json`
- `data/results/dinov3/{video_id}_dinov3.json`
- `data/results/tleap/{video_id}_tleap.json`

### Feature vector (current)

It produces a compact vector:

- **YOLO (4)**: confidence/stability/area/rate
- **SAM3 (3)**: area_ratio/circularity/aspect
- **DINOv3 (2)**: neighbor_evidence + similar_count
- **T‑LEAP (3)**: stride summary + head bob summary + asymmetry summary  
  (supports both legacy `locomotion_traits` and current `locomotion_features`)

### Model inference

- Loads trained models if present:
  - `/app/shared/models/ml/catboost_latest.cbm`
  - `/app/shared/models/ml/xgboost_latest.json`
  - `/app/shared/models/ml/lightgbm_latest.txt`
- If models exist:
  - predicts per-model probability
  - computes a weighted average using `ensemble_weights.json` or defaults
- Writes:
  - `data/results/ml/{video_id}_ml.json`
  - publishes `pipeline.ml`

---

## Fusion (concept)

Fusion consumes the outputs of per-video predictors and produces:

- `final_probability`
- `final_prediction`
- `confidence`
- per-pipeline contributions (used by the UI tabs)

The exact weighting/gating is implemented in `services/fusion-service/app/main.py`.

---

## Suggested reading order

1. `docs/ARCHITECTURE.md` (big picture)
2. `docs/PIPELINES_DETAILED.md` (this doc; step-by-step)
3. `docs/COW_POSE_DATA_GUIDE.md` (how to improve T‑LEAP pose quality)
4. `docs/ML_CONFIGURATION_GUIDE.md` (tuning/training the ML ensemble)


