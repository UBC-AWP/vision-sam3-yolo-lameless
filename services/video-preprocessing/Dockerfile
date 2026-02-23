FROM continuumio/miniconda3:latest

WORKDIR /app

# Auto-accept Anaconda Terms of Service for builds
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes

# Install system dependencies for OpenCV, FFmpeg, and awscli
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 ffmpeg awscli && rm -rf /var/lib/apt/lists/*

# Install mamba
RUN conda install -y mamba -n base -c conda-forge

# Copy environment file
COPY services/video-preprocessing/environment.yml /app/environment.yml

# Create conda environment
RUN mamba env create -f /app/environment.yml && \
    mamba clean -afy

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "video-preprocessing", "/bin/bash", "-c"]

# Copy application code
COPY services/video-preprocessing/ /app/
COPY shared/ /app/shared/

# Set environment variables (PYTHONUNBUFFERED so docker logs shows output)
ENV PATH /opt/conda/envs/video-preprocessing/bin:$PATH
ENV PYTHONPATH /app:$PYTHONPATH
ENV LONG_VIDEO_MODELS_DIR=/app/models
ENV PYTHONUNBUFFERED=1

# Run with env's python directly so docker logs gets output (conda run hides it)
CMD ["/opt/conda/envs/video-preprocessing/bin/python", "-u", "app/main.py"]

