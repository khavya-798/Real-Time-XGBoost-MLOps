# Dockerfile for Backend (FastAPI API - CORRECTED WITH CORRECT TRAINING SCRIPT)

# 1. Use a stable, MODERN base image
FROM python:3.10-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required by XGBoost
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY backend/requirements.txt /app/requirements.txt

# Install all dependencies (ensure training dependencies are included if different)
# NOTE: If local_train.py needs libraries not in backend/requirements.txt,
# you might need to copy and install from notebooks/requirements.txt too,
# or merge the requirements.
RUN pip install --no-cache-dir -r /app/requirements.txt

# --- ADD TRAINING STEPS ---
# Copy necessary files for training
COPY data/Fraud.csv /app/data/Fraud.csv
COPY notebooks/local_train.py /app/local_train.py # <-- COPY CORRECT SCRIPT

# RUN the training script - This will create the /app/models folder INSIDE the image
RUN python /app/local_train.py # <-- RUN CORRECT SCRIPT

# Clean up data and script if not needed at runtime (optional)
# RUN rm /app/data/Fraud.csv
# RUN rm /app/local_train.py
# --- END TRAINING STEPS ---

# Copy the rest of the backend application code
COPY ./backend/ /app/
# NOTE: No need to copy ./models/ anymore, it's created by local_train.py run above

# Expose the port
EXPOSE 8000

# Define the startup command (Ensure main.py and app exist in /app)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]