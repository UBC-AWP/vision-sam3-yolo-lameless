# Lameness Detection ML Pipeline System

A comprehensive Docker-based ML/AI pipeline system for cow lameness detection using YOLO, SAM3, DINOv3, CatBoost/XGBoost/LightGBM/Ensemble, and graph-based models.

## Architecture

The system is built as a microservices architecture with Docker containers, using NATS for asynchronous messaging between services.

### Core Components

1. **Video Ingestion Service** (Python/FastAPI): Upload, validate, and store videos
2. **Video Preprocessing Service** (Python): Crop videos to show only cow using YOLO detection
3. **ML/AI Pipeline Services** (Python):
   - YOLO Detection Pipeline
   - SAM3 Segmentation Pipeline
   - DINOv3 Embedding Pipeline
   - T-LEAP Pose Estimation Pipeline
   - ML Pipeline (CatBoost, XGBoost, LightGBM, Ensemble)
   - GNN/Graph Transformer Pipeline (optional)
4. **Fusion Service** (Python): Combine predictions from multiple pipelines
5. **SHAP Explainability Service** (Python): Generate explanations for predictions
6. **Training Service** (Python): Orchestrate model training
7. **Admin Interface** (FastAPI + React): Web UI for video upload, review, and analysis

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
├── services/              # Microservices
│   ├── video-ingestion/
│   ├── video-preprocessing/
│   ├── yolo-pipeline/
│   ├── sam3-pipeline/
│   ├── dinov3-pipeline/
│   ├── tleap-pipeline/
│   ├── ml-pipeline/
│   ├── gnn-pipeline/
│   ├── fusion-service/
│   ├── shap-service/
│   ├── training-service/
│   └── admin-interface/
│       ├── backend/      # FastAPI
│       └── frontend/     # Vite + React + shadcn/ui
├── shared/               # Shared code and config
│   ├── models/
│   ├── utils/
│   └── config/
├── data/                 # Data storage
│   ├── videos/
│   ├── processed/
│   ├── training/
│   └── results/
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

