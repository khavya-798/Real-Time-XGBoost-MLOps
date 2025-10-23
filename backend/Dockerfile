# Dockerfile for Backend (FastAPI API - Using archive.zip Data)

# 1. Use a stable, MODERN base image
FROM python:3.10-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (including unzip for data extraction)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 unzip \ # <-- Added unzip
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better layer caching)
COPY backend/requirements.txt /app/requirements.txt

# 2. Install all dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# --- ADD TRAINING STEPS ---
# Copy necessary files for training into the container's /app directory
COPY data/archive.zip /app/data/archive.zip # <-- Copy the CORRECT ZIP file
COPY notebooks/local_train.py /app/local_train.py

# RUN the training script - This script now handles unzipping archive.zip
RUN python /app/local_train.py

# Clean up zip file and script if not needed at runtime (optional, saves image space)
RUN rm /app/data/archive.zip # <-- Clean up CORRECT ZIP after use
# RUN rm /app/local_train.py
# --- END TRAINING STEPS ---

# 3. Copy the rest of the backend application code into the working directory
COPY ./backend/ /app/

# Expose the port FastAPI runs on
EXPOSE 8000

# 4. Define the startup command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]