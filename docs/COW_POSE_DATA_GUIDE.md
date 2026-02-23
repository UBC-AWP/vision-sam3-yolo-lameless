# Cow Pose Data Collection Guide

## The Problem

The current pose estimation is inaccurate because:
1. **Generic animal dataset** - The Animal Pose Dataset has limited cow data
2. **Wrong keypoints** - 20 generic keypoints don't align with cow anatomy
3. **Not optimized for lameness** - We need specific points for gait analysis

## Solution: Custom Cow Pose Dataset

For accurate lameness detection, we need to collect cow-specific pose data with keypoints optimized for gait analysis.

### Recommended Keypoints (8 points)

Based on the T-LEAP paper and lameness research:

| Index | Name | Description | Importance for Lameness |
|-------|------|-------------|------------------------|
| 0 | head | Center of poll/head | Head bob detection |
| 1 | withers | Top of shoulders | Back arch measurement |
| 2 | back | Center of spine | Back curvature |
| 3 | hip | Hip bone (tuber coxae) | Hip movement asymmetry |
| 4 | tailhead | Base of tail | Tail carriage |
| 5 | front_hoof | Visible front hoof | Step length, stance |
| 6 | rear_hoof | Visible rear hoof | Tracking, stride |
| 7 | belly | Bottom of belly | Height reference |

### Skeleton Connections

```
head ─── withers ─── back ─── hip ─── tailhead
            │                  │
            │                  │
        front_hoof         rear_hoof
            │                  │
            └────── belly ─────┘
```

## Data Collection Methods

### Option 1: Annotate Your Own Videos (Recommended)

Use our interactive annotation tool:

```bash
# Annotate a video interactively
python scripts/collect_cow_pose_data.py your_video.mp4 --output data/cow_pose_custom

# Export to YOLO format when done
python scripts/collect_cow_pose_data.py your_video.mp4 --output data/cow_pose_custom --export
```

**Tool Controls:**
- **Left Click** - Mark current keypoint
- **Space/Enter** - Save frame and go to next
- **N** - Skip to next frame (every 5th)
- **P** - Previous frame
- **S** - Save all annotations
- **R** - Reset current frame
- **Q/ESC** - Quit

**Tips for Good Annotations:**
1. Annotate frames where cow is fully visible (side view)
2. Skip frames where cow is occluded
3. Aim for 200-500 annotated frames
4. Include various walking speeds/gaits
5. Include both left-facing and right-facing views

### Option 2: Use Existing Datasets

1. **Cow Pose Estimation Dataset** (12 keypoints)
   - Papers With Code: https://paperswithcode.com/dataset/cow-pose-estimation-dataset
   - COCO format, compatible with YOLOv8

2. **Cattle Side View Dataset** (Mendeley)
   - https://data.mendeley.com/datasets/h2s22wr5py/3
   - 72 cattle with body measurements

3. **DeepLabCut Model Zoo**
   - Pre-trained models for various animals
   - Can be adapted for cows

### Option 3: Synthetic Data (Advanced)

Use the Synthetic Cow pipeline:
- https://mohajeranilab.github.io/projects/1_project/
- Generates synthetic cow images from 3D models
- Can create unlimited training data

## Training Your Model

After collecting annotations:

```bash
# Train YOLOv8-Pose on your custom data
python scripts/train_cow_pose_model.py \
    --data data/cow_pose_custom/cow_pose.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640

# The trained model will be saved to:
# - runs/pose/cow_pose/weights/best.pt
# - data/models/cow_pose_lameness.pt
```

### Training Tips

1. **Start Small**: Begin with 100-200 annotated frames
2. **Augmentation**: The training script includes rotation, scaling, flipping
3. **Validation**: Use 20% of data for validation
4. **Monitor**: Watch pose mAP50 metric during training
5. **Fine-tune**: If accuracy is low, add more training data

### Expected Metrics

| Metric | Good | Excellent |
|--------|------|-----------|
| Box mAP50 | >0.90 | >0.95 |
| Pose mAP50 | >0.60 | >0.75 |
| Pose mAP50-95 | >0.40 | >0.55 |

## Integrating Your Model

Once trained, update the T-LEAP pipeline to use your model:

```python
# In services/tleap-pipeline/app/main.py
MODEL_PATH = "/app/data/models/cow_pose_lameness.pt"
```

Rebuild the service:
```bash
docker compose build tleap-pipeline
docker compose up -d tleap-pipeline
```

## Lameness Features Extracted

With accurate pose data, we can compute:

| Feature | Calculation | Normal | Lame |
|---------|-------------|--------|------|
| Back Arch | Angle at back keypoint | ~170° | <160° |
| Head Bob | Vertical head movement | <5% height | >10% height |
| Stride Length | Distance between hooves | Consistent | Variable |
| Tracking | Rear hoof placement | On front track | Short/wide |
| Speed | Movement between frames | Normal | Slow |

## Recommended Workflow

1. **Collect Videos**
   - Record cows walking (side view)
   - Include known lame and healthy cows
   - Aim for 10+ videos per category

2. **Annotate Data**
   - Use the annotation tool
   - 30-50 frames per video
   - Focus on clear walking sequences

3. **Train Model**
   - Start with 300+ annotated frames
   - Train for 100 epochs
   - Validate on held-out data

4. **Evaluate**
   - Test on new videos
   - Check keypoint accuracy visually
   - Measure lameness detection accuracy

5. **Iterate**
   - Add more data if needed
   - Focus on failure cases
   - Re-train with expanded dataset

## Resources

- **T-LEAP Paper**: papers/T-LEAP*.pdf
- **Lameness Detection Paper**: papers/Lameness detection*.pdf
- **Video Locomotion Traits**: papers/Video-based Automatic*.pdf
- **YOLOv8-Pose Docs**: https://docs.ultralytics.com/tasks/pose/
- **DeepLabCut**: https://github.com/DeepLabCut/DeepLabCut

## Example Commands

```bash
# 1. Extract frames from video for annotation
python scripts/collect_cow_pose_data.py video.mp4 -o data/cow_pose -e -i 10

# 2. Annotate frames interactively
python scripts/collect_cow_pose_data.py video.mp4 -o data/cow_pose

# 3. Export annotations to YOLO format
python scripts/collect_cow_pose_data.py video.mp4 -o data/cow_pose --export

# 4. Train model
python scripts/train_cow_pose_model.py --data data/cow_pose/cow_pose.yaml --epochs 100

# 5. Test model on new video
python scripts/train_cow_pose_model.py --test new_video.mp4 --weights data/models/cow_pose_lameness.pt --data data/cow_pose/cow_pose.yaml
```

