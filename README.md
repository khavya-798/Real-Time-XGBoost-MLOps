# Real-Time Mobile Money Fraud Detection with MLOps (Fraud.csv Dataset)

🚀 ## Project Overview

This project demonstrates a full end-to-end MLOps workflow for a real-time mobile money fraud detection system using the `Fraud.csv` dataset. It includes a trained XGBoost model, feature engineering, preprocessing (scaling and one-hot encoding), a backend API to serve the model, a frontend user interface for interaction, and is fully containerized with Docker for reproducibility and deployment.

The application allows a user to input transaction details (type, amount, balances, etc.) via a web interface and receive an instant prediction on whether the transaction is potentially fraudulent or legitimate.

---

## Key Features

* **Machine Learning Model**: Utilizes an XGBoost classifier trained on the `Fraud.csv` dataset, with class weighting (`scale_pos_weight`) to handle severe class imbalance and optimize for high recall.
* **Feature Engineering**: Creates additional features like `balance_error` and `drained_account` based on transaction details to potentially improve model performance.
* **Preprocessing**: Handles both numerical features (using `StandardScaler`) and categorical features (`type` column using `OneHotEncoder`) before feeding data to the model.
* **Real-Time API**: A robust backend API built with FastAPI serves the ML model and preprocessors, handling data validation and prediction requests.
* **Interactive UI**: A user-friendly frontend built with Streamlit provides input fields for transaction features.
* **Containerized Environment**: The entire application (backend and frontend) is containerized using Docker and orchestrated locally with Docker Compose, ensuring consistency.
* **Cloud Deployment**: Both backend and frontend services are deployed as independent web services on Render.

---

## 🛠️ Tech Stack

* **Machine Learning**: Python, Pandas, Scikit-learn, XGBoost, Joblib
* **Backend**: FastAPI, Uvicorn, Pydantic
* **Frontend**: Streamlit, Requests
* **Infrastructure & MLOps**: Docker, Docker Compose
* **Deployment**: Render
* **Version Control**: Git (& Git LFS for large dataset)

---

## ☁️ Live Deployment (Render)

* **Frontend User Interface**: [`https://YOUR_FRONTEND_URL.onrender.com`](https://YOUR_FRONTEND_URL.onrender.com) <-- Replace with your actual Render frontend URL
* **Backend API Docs**: [`https://YOUR_BACKEND_URL.onrender.com/docs`](https://YOUR_BACKEND_URL.onrender.com/docs) <-- Replace with your actual Render backend URL + /docs
* **Backend Health Check**: [`https://YOUR_BACKEND_URL.onrender.com/health`](https://YOUR_BACKEND_URL.onrender.com/health) <-- Replace with your actual Render backend URL + /health

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
    Ensure you download the large `fraud.csv` dataset file:
    ```bash
    git lfs pull
    ```
3.  (Optional but Recommended) **Run Local Training:**
    If you want to retrain the model locally (e.g., after code changes) instead of using the one baked into the Docker image, run the training script. Ensure you have a Python environment with dependencies from `backend/requirements.txt` installed.
    ```bash
    # Make sure you are in the project root directory
    python notebooks/local_train.py
    ```
    This will generate/update the files in the `/models` directory.
4.  **Build and Run with Docker Compose:**
    This single command will build the Docker images (if not already built) for both the frontend and backend using their respective Dockerfiles, and start the services defined in `docker-compose.yml`.
    ```bash
    docker compose up --build
    ```
    *(Use `docker compose up` without `--build` for subsequent runs if no code/dependencies have changed)*

**Access the Local Application:**
* **Frontend (Streamlit App)**: Open your browser and go to `http://localhost:8501`
* **Backend (API Docs)**: Access API documentation at `http://localhost:8000/docs`
* **Health Check**: `http://localhost:8000/health`

---

*(Optional: Add a screenshot of your Streamlit UI)*
