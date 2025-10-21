FROM python:3.10-slim

# Set working directory
# WORKDIR /app/backend # <-- This is where the container will work

# Copy requirements and install dependencies
# FIX: The context is the project root (.), so specify the path to the requirements file.
COPY backend/requirements.txt /app/backend/requirements.txt 

# RUN installation using the requirements file now at its destination path
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# FIX: The context is the project root (.), so specify the path to the Python files.
COPY backend/main.py /app/backend/
COPY backend/preprocess.py /app/backend/
COPY ../.env /app/.env

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]