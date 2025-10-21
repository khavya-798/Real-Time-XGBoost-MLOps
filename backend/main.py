import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse # Import required for redirect
from pydantic import BaseModel, Field
import numpy as np
import xgboost as xgb
from .preprocess import load_transformers, preprocess_transaction, TRANSFORMERS_PATH

load_dotenv()

# --- Configuration ---
MODEL_FILE = os.getenv("MODEL_FILE", "xgboost_fraud_model.json")
MODEL_PATH = f"/app/models/{MODEL_FILE}" 

# --- Pydantic Schema ---
class Transaction(BaseModel):
    step: int = Field(..., description="Map time step to a day (integer)")
    type: str = Field(..., description="Type of transaction (e.g., CASH_OUT, PAYMENT)")
    amount: float = Field(..., description="Transaction amount")
    nameOrig: str = Field(..., description="Customer who initiated the transaction")
    oldbalanceOrg: float = Field(..., description="Original balance before transaction")
    newbalanceOrig: float = Field(..., description="New balance after transaction")
    nameDest: str = Field(..., description="Customer who is the recipient of the transaction")
    oldbalanceDest: float = Field(..., description="Original balance at recipient before transaction")
    newbalanceDest: float = Field(..., description="New balance at recipient after transaction")
    isFlaggedFraud: int = Field(..., description="Indicates if the system flagged the transaction (0 or 1)")

# --- Global Model and State (Initialization MUST happen here) ---
MODEL = None
INITIALIZATION_ERROR = None
MODEL_FEATURE_NAMES = None

try:
    MODEL = xgb.XGBClassifier()
    MODEL.load_model(MODEL_PATH)
    MODEL_FEATURE_NAMES = MODEL.get_booster().feature_names
    
    if not load_transformers():
        raise Exception(f"Failed to load preprocessing transformers at {TRANSFORMERS_PATH}")
        
    print("Backend ready: Model and Preprocessors initialized successfully.")

except Exception as e:
    INITIALIZATION_ERROR = f"Initialization failed: {e}"
    MODEL = None
    print(f"FATAL ERROR during startup: {INITIALIZATION_ERROR}")
    
app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="MLOps-ready API for XGBoost-based fraud prediction."
)

# --- Endpoints ---

@app.get("/", include_in_schema=False) # Exclude this endpoint from documentation schema
def root_redirect():
    """Redirects the root URL to the interactive API documentation."""
    # We redirect to /docs for the Swagger UI interface
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    """Standard health check, returns model status."""
    return {
        "status": "ok", 
        "model_loaded": (MODEL is not None),
        "model_path": MODEL_PATH,
        "error_detail": INITIALIZATION_ERROR if MODEL is None else None
    }

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    """Prediction endpoint."""
    
    # Check if initialization was successful globally
    if MODEL is None:
        return {"error": "MLOps service is not initialized. Model files missing or corrupted."}
    
    transaction_dict = transaction.model_dump()
    
    try:
        # 1. Preprocess (load_transformers() inside preprocess.py handles its own caching/loading)
        data = preprocess_transaction(transaction_dict, MODEL_FEATURE_NAMES) 

        # 2. Make prediction
        prediction = MODEL.predict(data)[0]
        probability = MODEL.predict_proba(data)[0][1] 

        is_fraud = bool(prediction)
        
        return {
            "prediction": int(prediction),
            "is_fraud": is_fraud,
            "probability_fraud": float(probability),
            "message": "Fraud detected by XGBoost" if is_fraud else "Transaction is likely genuine"
        }

    except Exception as e:
        print(f"Prediction error during inference: {e}")
        return {"error": f"Prediction pipeline failed during inference: {type(e).__name__}"}

    
