# /backend/main.py - Corrected FastAPI App

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import xgboost as xgb # Import xgboost
import numpy as np

# --- Configuration & Model/Preprocessor Loading ---
MODEL_PATH = "models/xgboost_fraud_model.json"
PREPROC_PATH = "models/preprocessing_transformers.joblib"

MODEL = None
PREPROCESSORS = None
INITIALIZATION_ERROR = None

# Define feature columns (must match training script)
CAT_COLS = ['type']
ORG_NUM_COLS = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'step']
NEW_NUM_FEATURES = ['balance_error', 'drained_account']
ALL_NUM_COLS = ORG_NUM_COLS + NEW_NUM_FEATURES

try:
    # Load the XGBoost model
    MODEL = xgb.XGBClassifier()
    MODEL.load_model(MODEL_PATH)
    
    # Load the dictionary of preprocessors
    PREPROCESSORS = joblib.load(PREPROC_PATH)
    
    if 'scaler' not in PREPROCESSORS or 'encoder' not in PREPROCESSORS or 'feature_names' not in PREPROCESSORS:
        raise Exception("Preprocessors file is missing 'scaler', 'encoder', or 'feature_names'.")
        
    print("Backend ready: Model and Preprocessors initialized successfully.")

except Exception as e:
    INITIALIZATION_ERROR = f"Initialization failed: {e}"
    MODEL = None
    PREPROCESSORS = None
    print(f"FATAL ERROR during startup: {INITIALIZATION_ERROR}")

# --- Pydantic Input Schema (7 Fields) ---
class TransactionData(BaseModel):
    step: int = Field(..., example=1)
    type: str = Field(..., example='PAYMENT')
    amount: float = Field(..., example=181.0)
    oldbalanceOrg: float = Field(..., example=181.0)
    newbalanceOrig: float = Field(..., example=0.0)
    oldbalanceDest: float = Field(..., example=0.0)
    newbalanceDest: float = Field(..., example=0.0)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="MLOps-ready API for XGBoost-based fraud prediction."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---
@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirects the root URL to the interactive API documentation."""
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    """Standard health check, returns model status."""
    return {
        "status": "ok",
        "model_loaded": (MODEL is not None and PREPROCESSORS is not None),
        "model_path": MODEL_PATH,
        "preprocessor_path": PREPROC_PATH,
        "error_detail": INITIALIZATION_ERROR
    }

@app.post("/predict")
def predict_fraud(data: TransactionData):
    """Predicts fraud based on transaction details."""
    if MODEL is None or PREPROCESSORS is None:
        raise HTTPException(status_code=503, detail=f"Service not initialized: {INITIALIZATION_ERROR}")

    try:
        input_data_dict = data.model_dump()
        input_df = pd.DataFrame([input_data_dict])

        # 1. Feature Engineering (match training)
        input_df['balance_error'] = input_df['oldbalanceOrg'] - input_df['newbalanceOrig'] - input_df['amount']
        input_df['drained_account'] = (input_df['newbalanceOrig'] == 0).astype(int)

        # 2. Preprocessing (match training)
        scaler = PREPROCESSORS['scaler']
        encoder = PREPROCESSORS['encoder']
        feature_names_out = PREPROCESSORS['feature_names']
        
        input_cat = input_df[CAT_COLS]
        input_num = input_df[ALL_NUM_COLS] # Includes engineered features

        encoded_cat_array = encoder.transform(input_cat)
        encoded_cat_df = pd.DataFrame(encoded_cat_array, columns=encoder.get_feature_names_out(CAT_COLS), index=input_df.index)

        scaled_num_array = scaler.transform(input_num)
        scaled_num_df = pd.DataFrame(scaled_num_array, columns=ALL_NUM_COLS, index=input_df.index)

        # Combine features
        processed_df = pd.concat([scaled_num_df, encoded_cat_df], axis=1)

        # 3. Reindex columns to match training order (CRUCIAL)
        processed_df = processed_df.reindex(columns=feature_names_out, fill_value=0)

        # 4. Predict class and probability
        prediction = MODEL.predict(processed_df)
        proba_array = MODEL.predict_proba(processed_df)

        is_fraud_value = int(prediction[0])
        probability_value = float(proba_array[0][1]) # Get probability of fraud (class 1)

        return {
            "is_fraud": is_fraud_value, 
            "probability": probability_value # <-- Correct key for frontend
        }

    except Exception as e:
        print(f"Prediction Error: {e}") # Log the error
        raise HTTPException(status_code=500, detail=f"Error during prediction processing: {str(e)}")