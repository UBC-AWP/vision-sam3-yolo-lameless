# Installation Guide

Complete setup guide for installing the Cow Lameness Detection System on a fresh computer.

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 16 GB | 32 GB |
| **Disk Space** | 50 GB | 100 GB |
| **CPU** | 4 cores | 8+ cores |
| **GPU** | Optional | NVIDIA GPU with CUDA (for faster ML inference) |
| **OS** | macOS 12+, Ubuntu 20.04+, Windows 11 with WSL2 | macOS 14+ (Apple Silicon optimized) |

---

## Prerequisites

### 1. Install Docker Desktop

Docker is required to run all services in containers.

#### macOS

```bash
# Option A: Download from website
# Visit https://www.docker.com/products/docker-desktop
# Download and install Docker Desktop for Mac

# Option B: Using Homebrew
brew install --cask docker
```

#### Ubuntu/Linux

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group (avoids needing sudo)
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose plugin
sudo apt-get update
sudo apt-get install docker-compose-plugin
```

#### Windows

1. Enable WSL2: `wsl --install` in PowerShell (Admin)
2. Download Docker Desktop from https://www.docker.com/products/docker-desktop
3. During installation, select "Use WSL 2 instead of Hyper-V"

#### Verify Docker Installation

```bash
docker --version
# Expected: Docker version 24.x or higher

docker compose version
# Expected: Docker Compose version v2.x or higher

# Test Docker is running
docker run hello-world
```

---

### 2. Install Git

Git is needed to clone the repository.

#### macOS

```bash
# Git comes with Xcode Command Line Tools
xcode-select --install

# Or using Homebrew
brew install git
```

#### Ubuntu/Linux

```bash
sudo apt-get update
sudo apt-get install git
```

#### Windows

Download from https://git-scm.com/download/win or use:

```powershell
winget install Git.Git
```

#### Verify Git Installation

```bash
git --version
# Expected: git version 2.x or higher
```

---

### 3. (Optional) Install Ollama for Local LLM

If you want AI-powered explanations without OpenAI API, install Ollama for local LLM support.

#### macOS

```bash
# Using Homebrew
brew install ollama

# Or download from https://ollama.ai
```

#### Linux

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Windows

Download from https://ollama.ai/download

#### Verify & Start Ollama

```bash
# Start Ollama service
ollama serve &

# Pull a recommended model
ollama pull llama3.2:3b

# Verify
ollama list
```

---

## Installation Steps

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/your-org/vision-sam3-yolo-lameless.git

# Navigate to the project directory
cd vision-sam3-yolo-lameless
```

### Step 2: Configure Environment (Optional)

Copy and customize the environment file:

```bash
# Copy example environment file
cp env.example .env

# Edit .env if you need to customize ports or add API keys
nano .env  # or use your preferred editor
```

**Key environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `DEPLOY_HOST` | localhost | Host for all services |
| `FRONTEND_PORT` | 3000 | Admin UI port |
| `BACKEND_PORT` | 8000 | API port |
| `POSTGRES_PORT` | 5432 | Database port |
| `OPENAI_API_KEY` | (empty) | For AI explanations (optional) |
| `OLLAMA_HOST` | http://host.docker.internal:11434 | Local LLM host |
| `OLLAMA_MODEL` | llama3.2:3b | Local LLM model |

### Step 3: Run Deployment Script

```bash
# Make the script executable
chmod +x scripts/deploy.sh

# Run deployment (this takes 10-20 minutes on first run)
./scripts/deploy.sh
```

The script will:
- ✅ Create required data directories
- ✅ Build all Docker images (~15 services)
- ✅ Start all containers
- ✅ Initialize PostgreSQL database
- ✅ Create Qdrant vector collections
- ✅ Seed default users

#### Alternative: Manual Deployment

If you prefer manual control:

```bash
# 1. Build all images
docker compose build

# 2. Start infrastructure services first
docker compose up -d postgres nats qdrant

# 3. Wait for database to be ready (10-15 seconds)
sleep 15

# 4. Initialize database
docker compose exec postgres psql -U lameness_user -d lameness_db < scripts/init_db.sql

# 5. Initialize Qdrant collections
curl -X PUT "http://localhost:6333/collections/cow_embeddings" \
    -H "Content-Type: application/json" \
    -d '{"vectors": {"size": 768, "distance": "Cosine"}}'

curl -X PUT "http://localhost:6333/collections/video_embeddings" \
    -H "Content-Type: application/json" \
    -d '{"vectors": {"size": 768, "distance": "Cosine"}}'

# 6. Start all services
docker compose up -d
```

### Step 4: Verify Installation

```bash
# Check all containers are running
docker compose ps

# Expected: All services should show "running" status
```

Visit these URLs to verify:

| Service | URL | Expected |
|---------|-----|----------|
| **Frontend** | http://localhost:3000 | Login page |
| **API Docs** | http://localhost:8000/docs | Swagger UI |
| **NATS Monitor** | http://localhost:8222 | NATS stats |
| **Qdrant Dashboard** | http://localhost:6333/dashboard | Vector DB UI |

---

## First Login

### Default Credentials

| Role | Email | Password |
|------|-------|----------|
| **Admin** | admin@example.com | adminpass123 |
| **Researcher** | researcher@example.com | researcher123 |
| **Rater** | rater@example.com | rater123 |

1. Open http://localhost:3000
2. Login with `admin@example.com` / `adminpass123`
3. Navigate through the dashboard

> ⚠️ **Security Note:** Change the default admin password immediately in Settings → User Profile.

---

## Post-Installation Setup

### Upload Sample Videos

1. Navigate to **Video Analysis** in the sidebar
2. Click **Upload Video**
3. Select a cow walking video (MP4 format recommended)
4. Wait for the pipeline to process

### Configure Theme

1. Go to **Settings** in the sidebar
2. Select **Appearance** tab
3. Choose your preferred theme (Light, Dark, Midnight, Ocean, etc.)

### Check Pipeline Health

1. Navigate to **System Health**
2. Verify all services show "Healthy" status
3. Check **Pipeline Monitor** for real-time processing

---

## Troubleshooting

### Docker Issues

**Issue: "Cannot connect to Docker daemon"**

```bash
# macOS/Windows: Start Docker Desktop application
# Linux: Start Docker service
sudo systemctl start docker
```

**Issue: Port already in use**

```bash
# Find what's using the port
lsof -i :3000

# Change ports in .env file or stop the conflicting service
```

**Issue: Out of disk space**

```bash
# Clean up Docker resources
docker system prune -a

# Remove unused volumes (⚠️ deletes data!)
docker volume prune
```

### Database Issues

**Issue: "Database connection refused"**

```bash
# Check if postgres is running
docker compose ps postgres

# Restart postgres
docker compose restart postgres

# Wait 10 seconds, then try again
```

**Issue: "Column does not exist"**

```bash
# Re-run database initialization
docker compose exec postgres psql -U lameness_user -d lameness_db < scripts/init_db.sql
```

### Service Issues

**Check logs for a specific service:**

```bash
docker compose logs <service-name>

# Examples:
docker compose logs admin-frontend
docker compose logs ml-pipeline
docker compose logs fusion-service
```

**Restart a single service:**

```bash
docker compose restart <service-name>
```

**Rebuild and restart a service:**

```bash
docker compose build <service-name>
docker compose up -d <service-name>
```

### Complete Reset

If you need to start fresh:

```bash
# Stop all services and remove volumes
docker compose down -v

# Remove all images
docker compose down --rmi all

# Clean deployment
./scripts/deploy.sh --clean
```

---

## Updating the System

### Pull Latest Code

```bash
# Pull latest changes
git pull origin main

# Rebuild changed services
docker compose build

# Restart services
docker compose down
docker compose up -d

# Run migrations if schema changed
docker compose exec postgres psql -U lameness_user -d lameness_db < scripts/init_db.sql
```

### Update Single Service

```bash
docker compose build <service-name>
docker compose up -d <service-name>
```

---

## Development Setup (Optional)

For local development without Docker:

### Install Conda/Mamba

```bash
# Install Miniforge (recommended)
# macOS
brew install miniforge
conda init zsh  # or bash

# Linux
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

### Create Development Environment

```bash
# Create base environment
conda env create -f environment.yml
conda activate lameness-detection-base

# For a specific service
cd services/ml-pipeline
conda env create -f environment.yml
conda activate ml-pipeline
```

### Run Service Locally

```bash
# Start infrastructure only
docker compose up -d postgres nats qdrant

# Run a service locally
cd services/admin-interface/backend
conda activate admin-backend
python -m uvicorn app.main:app --reload --port 8000
```

---

## Next Steps

After installation:

1. 📖 Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) to understand the system
2. 📊 Upload sample videos and test the pipeline
3. 🎨 Customize your theme in Settings
4. 🔒 Change default passwords
5. 📚 Review [docs/PIPELINES_DETAILED.md](docs/PIPELINES_DETAILED.md) for ML details

---

## Support

- Check [DEPLOYMENT.md](DEPLOYMENT.md) for deployment details
- Review [docs/](docs/) for architecture documentation
- Check container logs: `docker compose logs <service>`


