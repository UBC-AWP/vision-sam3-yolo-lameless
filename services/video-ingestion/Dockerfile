FROM continuumio/miniconda3:latest

WORKDIR /app

# Auto-accept Anaconda Terms of Service for builds
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes

# Install mamba for faster conda operations
RUN conda install -y mamba -n base -c conda-forge

# Copy environment file
COPY services/video-ingestion/environment.yml /app/environment.yml

# Create conda environment
RUN mamba env create -f /app/environment.yml && \
    mamba clean -afy

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "video-ingestion", "/bin/bash", "-c"]

# Copy application code
COPY services/video-ingestion/ /app/
COPY shared/ /app/shared/

# Set environment variables
ENV PATH /opt/conda/envs/video-ingestion/bin:$PATH
ENV PYTHONPATH /app:$PYTHONPATH

# Expose port
EXPOSE 8000

# Run the application
CMD ["conda", "run", "-n", "video-ingestion", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

