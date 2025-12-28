# Lameness Detection ML Pipeline System

A comprehensive Docker-based ML/AI pipeline system for cow lameness detection using YOLO, SAM3, DINOv3, CatBoost/XGBoost/LightGBM/Ensemble, and graph-based models.

## Architecture

The system is built as a microservices architecture with Docker containers, using NATS for asynchronous messaging between services.

### System Diagram

![System Architecture](docs/design-diagram.png)

> For detailed architecture documentation, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

### Core Components

1. **Video Processing Layer**:
   - Video Ingestion Service - Upload, validate, and store videos
   - Video Preprocessing Service - Crop videos using YOLO detection
   - Clip Curation Service - Extract optimal 5s canonical clips

2. **Feature Extraction Pipelines**:
   - YOLO Detection Pipeline - Bounding boxes, confidence
   - SAM3 Segmentation Pipeline - Silhouette masks
   - DINOv3 Embedding Pipeline - 768-dim feature vectors
   - T-LEAP Pose Pipeline - Keypoints, locomotion metrics

3. **Deep Learning Pipelines**:
   - TCN Pipeline - Temporal Convolutional Network for gait analysis
   - Transformer Pipeline - Self-attention based temporal modeling
   - GraphGPS Pipeline - Graph Transformer for relational context

4. **ML Ensemble**: CatBoost, XGBoost, LightGBM with stacking

5. **Human-in-the-Loop**:
   - Pairwise Comparison (7-point scale)
   - Triplet Comparison (similarity/dissimilarity)
   - Rater Reliability (Dawid-Skene, tier system)

6. **Fusion & Explainability**:
   - Fusion Service - Combine all predictions with gating rules
   - SHAP Service - Feature importance explanations
   - LLM Service - Natural language summaries

7. **Admin Interface** (FastAPI + React): Dashboard, visualization, training module

## Prerequisites

- Docker and Docker Compose
- Conda/Mamba (for local development)

## Quick Start

1. **Clone the repository** (if not already done)

2. **Start all services with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

3. **Access the admin interface:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## Development Setup

### Using Conda

1. **Create base conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate lameness-detection-base
   ```

2. **Create service-specific environments:**
   ```bash
   cd services/video-ingestion
   conda env create -f environment.yml
   conda activate video-ingestion
   ```

### Running Services Locally

Each service can be run independently:

```bash
cd services/video-ingestion
conda activate video-ingestion
python -m uvicorn app.main:app --reload --port 8001
```

## Project Structure

```
vision-sam3-yolo-lameless/
├── services/                    # 22 Microservices
│   ├── video-ingestion/         # Upload handling
│   ├── video-preprocessing/     # YOLO-based cropping
│   ├── clip-curation/           # 5s canonical clip extraction
│   ├── yolo-pipeline/           # Object detection
│   ├── sam3-pipeline/           # Segmentation
│   ├── dinov3-pipeline/         # Embeddings
│   ├── tleap-pipeline/          # Pose estimation
│   ├── tcn-pipeline/            # Temporal CNN
│   ├── transformer-pipeline/    # Gait Transformer
│   ├── gnn-pipeline/            # GraphGPS
│   ├── ml-pipeline/             # XGBoost/CatBoost/LightGBM
│   ├── fusion-service/          # Prediction fusion
│   ├── rater-reliability/       # Dawid-Skene/GLAD
│   ├── shap-service/            # Explainability
│   ├── llm-service/             # Natural language explanations
│   ├── training-service/        # Model training orchestration
│   ├── annotation-renderer/     # Video annotation overlay
│   └── admin-interface/
│       ├── backend/             # FastAPI REST API
│       └── frontend/            # React + TypeScript + Tailwind
├── shared/                      # Shared code and config
│   ├── models/
│   ├── utils/
│   └── config/
├── data/                        # Data storage
│   ├── videos/
│   ├── canonical/               # Curated 5s clips
│   ├── processed/
│   ├── training/
│   ├── results/
│   └── quality_reports/
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md
│   └── design-diagram.png
├── research/                    # Research code and papers
├── docker-compose.yml
├── environment.yml
└── README.md
```

## API Endpoints

See the FastAPI documentation at http://localhost:8000/docs for complete API reference.

## Training Models

See [TRAINING.md](TRAINING.md) for detailed instructions on training YOLO, ML models, and ensemble methods.

## License

See LICENSE file for details.

