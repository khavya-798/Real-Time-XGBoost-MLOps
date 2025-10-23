# Real-Time Mobile Money Fraud Detection with MLOps

🚀 ## Project Overview

This project demonstrates a full end-to-end MLOps workflow for a real-time mobile money fraud detection system. It's built using the `Fraud.csv` dataset, containerized with Docker, and deployed on Render.

The application features:
* A **FastAPI backend API** that serves a trained XGBoost model.
* A **Streamlit frontend UI** that allows users to input transaction data and receive instant fraud predictions.

---

## Key Features

* **Machine Learning Model**: Utilizes an XGBoost classifier trained on the `Fraud.csv` dataset. It uses class weighting (`scale_pos_weight`) to handle the severe class imbalance and optimize for high recall.
* **Feature Engineering**: The training pipeline automatically creates features like `balance_error` and `drained_account` from raw transaction data.
* **Preprocessing**: Handles mixed data types by applying `StandardScaler` to numerical features and `OneHotEncoder` to categorical features (like `type`).
* **Automated Build & Training**: The backend's Dockerfile is configured to download the `archive.zip` data file directly from cloud storage (Google Drive) using `gdown` during the Render build, bypassing Git LFS limits. It then runs the training script (`local_train.py`) to build the model artifacts (`.json`, `.joblib`) *inside* the image.
* **Cloud Deployment**: Both the backend API and frontend UI are deployed as independent, communicating web services on Render.

---

## 🛠️ Tech Stack

* **Machine Learning**: Python, Pandas, Scikit-learn, XGBoost, Joblib
* **Backend**: FastAPI, Uvicorn, Pydantic, Gdown
* **Frontend**: Streamlit, Requests
* **Infrastructure & MLOps**: Docker, Docker Compose
* **Deployment**: Render
* **Version Control**: Git, Git LFS

---

## ☁️ Live Deployment (Render)

* **Frontend User Interface**: [`https://fraud-frontend-ui.onrender.com`](https://fraud-frontend-ui.onrender.com)
* **Backend API Docs**: [`https://real-time-xgboost-mlops.onrender.com/docs`](https://real-time-xgboost-mlops.onrender.com/docs)
* **Backend Health Check**: [`https://real-time-xgboost-mlops.onrender.com/health`](https://real-time-xgboost-mlops.onrender.com/health)

*(Note: Free Render services may spin down after inactivity and take ~30-60 seconds to start on the first request.)*

---

## ⚙️ Local Setup and Installation

To run this project on your local machine for development or testing:

**Prerequisites:**
* Docker installed on your machine.
* Docker Compose V2 (usually included with Docker Desktop).
* Git installed.
* Git LFS installed (`git lfs install`).

**Installation Steps:**
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/khavya-798/Real-Time-XGBoost-MLOps.git](https://github.com/khavya-798/Real-Time-XGBoost-MLOps.git)
    cd Real-Time-XGBoost-MLOps
    ```
2.  **Pull LFS files:**
    (This step may not be necessary if `archive.zip` is no longer tracked by LFS, but it's good practice for other potential LFS files.)
    ```bash
    git lfs pull
    ```
3.  **(Optional) Run Local Training:**
    If you want to train the model locally *before* building the Docker image, place `archive.zip` in the `data/` folder and run the training script. Ensure you have a Python environment with dependencies from `backend/requirements.txt` installed.
    ```bash
    # Make sure you are in the project root directory
    python notebooks/local_train.py
    ```
    This will generate the files in the `/models` directory, which can then be used by the local API.

4.  **Build and Run with Docker Compose:**
    This single command will build the Docker images and start the services. The backend build process will download the data and run the training script automatically as defined in its Dockerfile.
    ```bash
    docker compose up --build
    ```
    *(Use `docker compose up` without `--build` for subsequent runs if no code/dependencies have changed)*

**Access the Local Application:**
* **Frontend (Streamlit App)**: Open your browser and go to `http://localhost:8501`
* **Backend (API Docs)**: Access API documentation at `http://localhost:8000/docs`
* **Health Check**: `http://localhost:8000/health`
