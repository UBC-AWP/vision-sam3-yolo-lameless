# Deployment Guide

## Infrastructure Requirements

### Services Started by Docker Compose

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| **postgres** | postgres:15-alpine | 5432 | Main database |
| **nats** | nats:2.10 | 4222, 8222 | Message broker |
| **qdrant** | qdrant/qdrant | 6333, 6334 | Vector database |

### Application Services

| Service | Purpose | Dependencies |
|---------|---------|--------------|
| video-ingestion | Upload & validate videos | nats, postgres |
| video-preprocessing | YOLO-based cropping | nats |
| clip-curation | Extract canonical clips | nats |
| yolo-pipeline | Object detection | nats |
| sam3-pipeline | Segmentation masks | nats |
| dinov3-pipeline | Visual embeddings | nats |
| tleap-pipeline | Pose estimation | nats |
| tcn-pipeline | Temporal CNN | nats |
| transformer-pipeline | Attention-based gait | nats |
| gnn-pipeline | GraphGPS | nats |
| graph-transformer-pipeline | Graphormer | nats |
| tracking-service | Cow ID tracking | nats, postgres, qdrant |
| ml-pipeline | ML ensemble | nats |
| fusion-service | Prediction fusion | nats |
| shap-service | Explainability | nats |
| admin-backend | REST API | postgres |
| admin-frontend | React UI | admin-backend |

## Fresh Deployment Steps

### 1. Run Deployment Script

```bash
./scripts/deploy.sh
```

Or for a clean start:

```bash
./scripts/deploy.sh --clean
```

### 2. Manual Steps (if needed)

#### Initialize Database

```bash
# Connect to postgres and run init script
docker compose exec postgres psql -U lameness_user -d lameness_db < scripts/init_db.sql
```

#### Initialize Qdrant Collections

```bash
# Create cow_embeddings collection (for Re-ID)
curl -X PUT "http://localhost:6333/collections/cow_embeddings" \
    -H "Content-Type: application/json" \
    -d '{"vectors": {"size": 768, "distance": "Cosine"}}'

# Create video_embeddings collection
curl -X PUT "http://localhost:6333/collections/video_embeddings" \
    -H "Content-Type: application/json" \
    -d '{"vectors": {"size": 768, "distance": "Cosine"}}'
```

## Database Schema

The database is managed via SQLAlchemy ORM with auto-creation on startup. For schema changes:

1. Update models in `services/admin-interface/backend/app/database.py`
2. Add migration SQL to `scripts/init_db.sql`
3. Run the init script on existing databases

### Tables

| Table | Purpose |
|-------|---------|
| users | User accounts & auth |
| sessions | JWT session management |
| processing_jobs | Video processing queue |
| gold_tasks | Rater validation tasks |
| rater_stats | Rater performance metrics |
| video_elo_ratings | Video lameness rankings |
| pairwise_comparisons | Human comparison records |
| elo_history | Rating history snapshots |
| hierarchy_snapshots | Full ranking exports |
| service_heartbeats | Service health monitoring |
| cow_identities | Known cow registry |
| track_history | Video-to-cow mappings |
| lameness_records | Per-cow lameness scores |

## Data Directories

```
data/
├── videos/           # Raw uploaded videos
├── canonical/        # Curated 5-second clips
├── processed/        # Preprocessed frames
├── training/         # Training datasets
├── quality_reports/  # Video quality assessments
└── results/          # Pipeline outputs
    ├── yolo/
    ├── sam3/
    ├── dinov3/
    ├── tleap/
    ├── tcn/
    ├── transformer/
    ├── gnn/
    ├── graph_transformer/
    ├── ml/
    ├── fusion/
    ├── tracking/
    ├── shap/
    └── cow_predictions/
```

## Environment Variables

Key environment variables (set in docker-compose.yml):

```bash
# Database
POSTGRES_URL=postgresql://lameness_user:lameness_pass@postgres:5432/lameness_db

# NATS
NATS_URL=nats://nats:4222

# Qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# Data paths
DATA_DIR=/app/data
```

## Default Seed Data

### Users

| Email | Password | Role | Tier |
|-------|----------|------|------|
| admin@example.com | adminpass123 | admin | gold |
| researcher@example.com | researcher123 | researcher | gold |
| rater@example.com | rater123 | rater | bronze |

### Service Heartbeats

All 15 pipeline services are pre-registered in `service_heartbeats` for monitoring.

## Troubleshooting

### Database column missing

```bash
# Run init script to add missing columns
docker compose exec postgres psql -U lameness_user -d lameness_db < scripts/init_db.sql
```

### Service not starting

```bash
# Check logs
docker compose logs <service-name>

# Rebuild specific service
docker compose build <service-name>
docker compose up -d <service-name>
```

### Reset everything

```bash
./scripts/deploy.sh --clean
```

## Production Considerations

1. **Database backups**: Set up automated PostgreSQL backups
2. **HTTPS**: Use reverse proxy (nginx/traefik) with SSL
3. **Secrets**: Use Docker secrets or external secret manager
4. **Scaling**: Consider Kubernetes for horizontal scaling
5. **Monitoring**: Add Prometheus/Grafana for observability

