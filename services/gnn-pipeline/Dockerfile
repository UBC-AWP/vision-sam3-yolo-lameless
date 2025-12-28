FROM continuumio/miniconda3:latest

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install mamba
RUN conda install -y mamba -n base -c conda-forge

# Copy environment file
COPY services/gnn-pipeline/environment.yml /app/environment.yml

# Create conda environment
RUN mamba env create -f /app/environment.yml && \
    mamba clean -afy

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "gnn-pipeline", "/bin/bash", "-c"]

# Copy application code
COPY services/gnn-pipeline/app /app/app
COPY shared /app/shared

# Set environment variables
ENV PATH /opt/conda/envs/gnn-pipeline/bin:$PATH
ENV PYTHONPATH /app:$PYTHONPATH

# Run the application
CMD ["conda", "run", "-n", "gnn-pipeline", "python", "-m", "app.main"]
