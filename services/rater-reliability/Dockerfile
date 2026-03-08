FROM continuumio/miniconda3:24.1.2-0

WORKDIR /app

# Install mamba
RUN conda install -y mamba -n base -c conda-forge

# Copy environment file
COPY services/rater-reliability/environment.yml /app/environment.yml

# Create conda environment (-y non-interactive; only conda-forge in env avoids channel conflicts)
RUN mamba env create -f /app/environment.yml -y && \
    mamba clean -afy

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "rater-reliability", "/bin/bash", "-c"]

# Copy application code
COPY services/rater-reliability/app /app/app
COPY shared /app/shared

# Set environment variables
ENV PATH /opt/conda/envs/rater-reliability/bin:$PATH
ENV PYTHONPATH /app:$PYTHONPATH

# Run the application
CMD ["conda", "run", "-n", "rater-reliability", "python", "-m", "app.main"]
