FROM continuumio/miniconda3:latest

WORKDIR /app

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

# Set environment variables
ENV PATH /opt/conda/envs/video-preprocessing/bin:$PATH

# Run the application
CMD ["conda", "run", "-n", "video-preprocessing", "python", "app/main.py"]

