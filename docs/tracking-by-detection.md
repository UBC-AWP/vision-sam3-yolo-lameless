# Tracking-by-detection (as implemented in this repo)

## What this service is for

The goal of tracking in this system is **cow identity**:

- **Within a video**: keep a stable track id for a cow while it walks
- **Across videos**: match the same cow again later (persistent `cow_id`)

In this repo, that is implemented by `tracking-service`.

---

## Core idea (tracking-by-detection)

1. **Detect** cows in each frame (YOLO produces bounding boxes)
2. **Associate** detections across frames into tracks (ByteTrack-style association)
3. **Re‑ID** the track across videos using appearance embeddings (DINOv3 + Qdrant)

This split is robust in farm videos because:
- multiple cows can look similar
- occlusions happen often
- the walking direction and framing vary between clips

---

## Repo-specific integration

### Inputs

The tracking service consumes:

- `pipeline.yolo` (detections by frame)
- `pipeline.dinov3` (appearance embeddings; used for Re‑ID)

NATS subjects are defined in `shared/config/config.yaml`.

### Outputs

Tracking produces:

- `data/results/tracking/{video_id}_tracking.json`
- Postgres updates:
  - `cow_identities` (persistent cows)
  - `track_history` (video sightings)
- NATS publishes:
  - `tracking.complete`
  - `tracking.reid.match`

---

## Algorithms used (practical)

### Within-video tracking (ByteTrack-like)

Typical steps:

- Keep a set of active tracks
- For each new frame:
  - predict track positions (Kalman filter)
  - compute association cost (IoU and/or distance)
  - match detections to tracks (Hungarian assignment)
  - update matched tracks; create new tracks for unmatched detections

### Cross-video Re‑ID (vector DB)

This repo uses a vector DB approach for persistence:

- Use an embedding model (DINOv3 pipeline) to produce a vector representation
- Store/query vectors in **Qdrant**
- Match new tracks to existing cows by similarity (Cosine distance)

This makes identity “sticky” across days and different videos.

---

## How this affects lameness prediction

Once a clip is assigned to a cow:

- the UI can show **Cow Registry** (cow → list of videos)
- graph-based predictors can build **per-cow graphs** (optional/next step depending on configuration)
- the system can write to `lameness_records` to build cow-level history
