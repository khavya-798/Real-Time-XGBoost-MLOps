# Dockerfile for Backend (FastAPI API - CORRECTED WITH TRAINING STEP)

# 1. Use a stable, MODERN base image
FROM python:3.10-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required by XGBoost (OpenMP runtime)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better layer caching)
COPY backend/requirements.txt /app/requirements.txt

# 2. Install all dependencies
# Ensure backend/requirements.txt includes pandas, scikit-learn, xgboost, joblib if local_train.py needs them
RUN pip install --no-cache-dir -r /app/requirements.txt

# --- ADD TRAINING STEPS ---
# Copy necessary files for training into the container's /app directory
COPY data/fraud.csv /app/data/fraud.csv
COPY notebooks/local_train.py /app/local_train.py

# RUN the training script - This will create the /app/models folder INSIDE the image
# Make sure local_train.py uses paths relative to /app (e.g., 'data/fraud.csv', 'models')
RUN python /app/local_train.py

# Clean up data and script if not needed at runtime (optional, saves image space)
# RUN rm /app/data/fraud.csv
# RUN rm /app/local_train.py
# --- END TRAINING STEPS ---

# 3. Copy the rest of the backend application code into the working directory
# This includes main.py (or app.py) which loads the models created above
COPY ./backend/ /app/

# Expose the port FastAPI runs on
EXPOSE 8000

# 4. Define the startup command (runs from /app where main.py now is)
# Ensure 'main:app' matches your filename and FastAPI instance name
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]