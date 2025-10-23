# Dockerfile for Backend (FastAPI API - Downloading Zipped Data)

# 1. Use a stable, MODERN base image
FROM python:3.10-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (including curl/wget and unzip)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 unzip curl \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better layer caching)
COPY backend/requirements.txt /app/requirements.txt

# 2. Install all dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# --- ADD TRAINING STEPS ---
# Create data directory
RUN mkdir -p /app/data

# Download the zipped data file (Using your Google Drive Link)
RUN curl -L -o /app/data/archive.zip "https://drive.google.com/uc?export=download&id=1l7dtEqO4SM-k0kzi-uyZ8QPyBPo7CjHW"

# Copy the training script
COPY notebooks/local_train.py /app/local_train.py

# RUN the training script - This script handles unzipping archive.zip
RUN python /app/local_train.py

# Clean up zip file and script if not needed at runtime (optional, saves image space)
RUN rm /app/data/archive.zip
# RUN rm /app/local_train.py
# --- END TRAINING STEPS ---

# 3. Copy the rest of the backend application code into the working directory
# This includes main.py which loads the models created above
COPY ./backend/ /app/

# Expose the port FastAPI runs on
EXPOSE 8000

# 4. Define the startup command
# Ensure 'main:app' matches your filename and FastAPI instance name
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]