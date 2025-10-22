# Dockerfile for Backend (FastAPI API - CORRECTED)

# 1. Use a stable, MODERN base image
FROM python:3.10-slim-bullseye

# Set the working directory inside the container
WORKDIR /app  # <--- UNCOMMENTED and set to /app

# Install system dependencies required by XGBoost (OpenMP runtime)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Copy the backend-specific requirements file first (for layer caching)
COPY backend/requirements.txt /app/requirements.txt

# 2. Install all dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# 3. Copy the backend application code into the working directory
COPY ./backend/ /app/
# Copy models required for the application (baked into the image for Render)
COPY ./models/ /app/models/

# Copy .env file if it's in the project root and needed at runtime
# Consider using Render's Environment Variables instead for secrets/config
# COPY .env /app/.env

# Expose the port (FastAPI default)
EXPOSE 8000

# 4. Define the startup command (runs from /app where main.py now is)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]