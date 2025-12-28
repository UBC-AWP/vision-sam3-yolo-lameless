FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY services/gnn-pipeline/environment.yml .
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    torchvision>=0.15.0 \
    torch-geometric>=2.4.0 \
    torch-scatter \
    torch-sparse \
    nats-py>=2.6.0 \
    pyyaml>=6.0.1 \
    numpy>=1.24.0

# Copy application code
COPY services/gnn-pipeline/app /app/app
COPY shared /app/shared

# Set Python path
ENV PYTHONPATH=/app

CMD ["python", "-m", "app.main"]

