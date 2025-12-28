FROM continuumio/miniconda3:latest

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Install mamba
RUN conda install -y mamba -n base -c conda-forge

# Copy environment file
COPY services/sam3-pipeline/environment.yml /app/environment.yml

# Create conda environment
RUN mamba env create -f /app/environment.yml && \
    mamba clean -afy

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "sam3-pipeline", "/bin/bash", "-c"]

# Copy application code
COPY services/sam3-pipeline/ /app/
COPY shared/ /app/shared/

# Set environment variables
ENV PATH /opt/conda/envs/sam3-pipeline/bin:$PATH
ENV PYTHONPATH /app:$PYTHONPATH

# Run the application
CMD ["conda", "run", "-n", "sam3-pipeline", "python", "app/main.py"]

