FROM python:3.10-slim

WORKDIR /app

# Install Python dependencies
COPY services/llm-service/environment.yml .
RUN pip install --no-cache-dir \
    openai>=1.0.0 \
    nats-py>=2.6.0 \
    pyyaml>=6.0.1

# Copy application code
COPY services/llm-service/app /app/app
COPY shared /app/shared

# Set Python path
ENV PYTHONPATH=/app

CMD ["python", "-m", "app.main"]

