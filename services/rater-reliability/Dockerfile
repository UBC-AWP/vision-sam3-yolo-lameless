FROM python:3.10-slim

WORKDIR /app

# Install Python dependencies
COPY services/rater-reliability/environment.yml .
RUN pip install --no-cache-dir \
    nats-py>=2.6.0 \
    pyyaml>=6.0.1 \
    numpy>=1.24.0

# Copy application code
COPY services/rater-reliability/app /app/app
COPY shared /app/shared

# Set Python path
ENV PYTHONPATH=/app

CMD ["python", "-m", "app.main"]

