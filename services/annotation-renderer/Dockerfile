FROM continuumio/miniconda3:latest

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install mamba for faster package resolution
RUN conda install -y mamba -n base -c conda-forge

# Copy environment file
COPY services/annotation-renderer/environment.yml /app/environment.yml

# Create conda environment
RUN mamba env create -f /app/environment.yml && \
    mamba clean -afy

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "annotation-renderer", "/bin/bash", "-c"]

# Copy application code
COPY services/annotation-renderer/ /app/
COPY shared/ /app/shared/

# Set environment variables
ENV PATH /opt/conda/envs/annotation-renderer/bin:$PATH
ENV PYTHONPATH /app:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["conda", "run", "--no-capture-output", "-n", "annotation-renderer", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

