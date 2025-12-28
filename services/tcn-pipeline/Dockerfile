FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY services/tcn-pipeline/environment.yml .
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    torchvision>=0.15.0 \
    nats-py>=2.6.0 \
    pyyaml>=6.0.1 \
    numpy>=1.24.0

# Copy application code
COPY services/tcn-pipeline/app /app/app
COPY shared /app/shared

# Set Python path
ENV PYTHONPATH=/app

CMD ["python", "-m", "app.main"]

