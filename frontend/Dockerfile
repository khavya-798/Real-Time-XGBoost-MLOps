# Dockerfile for frontend (Streamlit)

FROM python:3.10-slim

# Set working directory to the container's app path
# All subsequent paths will be relative to /app inside the container
WORKDIR /app

# 1. Copy requirements and install dependencies
# FIX: Copy from the root context (frontend/requirements.txt) 
# to the desired location inside the container (/app/frontend/requirements.txt)
COPY frontend/requirements.txt /app/frontend/requirements.txt
RUN pip install --no-cache-dir -r /app/frontend/requirements.txt

# 2. Copy application code and data
# Copy frontend/app.py to /app/frontend/
COPY frontend/app.py /app/frontend/
# Copy the dataset from the data folder to a known location inside the container
COPY data/fraud.csv /app/data/fraud.csv

COPY frontend/style.css /app/style.css
# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit application
# The path must be relative to the WORKDIR (/app)
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]