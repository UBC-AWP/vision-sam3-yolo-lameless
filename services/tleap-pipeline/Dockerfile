FROM continuumio/miniconda3:latest

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Install mamba
RUN conda install -y mamba -n base -c conda-forge

# Copy environment file
COPY services/tleap-pipeline/environment.yml /app/environment.yml

# Create conda environment
RUN mamba env create -f /app/environment.yml && \
    mamba clean -afy

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "tleap-pipeline", "/bin/bash", "-c"]

# Copy application code
COPY services/tleap-pipeline/ /app/
COPY shared/ /app/shared/

# Set environment variables
ENV PATH /opt/conda/envs/tleap-pipeline/bin:$PATH
ENV PYTHONPATH /app:$PYTHONPATH
ENV PYTHONUNBUFFERED 1

# Run the application
CMD ["conda", "run", "--no-capture-output", "-n", "tleap-pipeline", "python", "app/main.py"]

