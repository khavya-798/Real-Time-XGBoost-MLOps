# preprocess.py - Contains all preprocessing logic from the Kaggle notebook

import pandas as pd
import numpy as np
import joblib
import os
from dotenv import load_dotenv

load_dotenv()
TRANSFORMERS_FILE = "preprocessing_transformers.joblib"
# preprocess.py - Contains all preprocessing logic from the Kaggle notebook

# ... (lines 1-10)
TRANSFORMERS_FILE = "preprocessing_transformers.joblib"
# CHANGE THIS LINE: Use absolute path inside the container
TRANSFORMERS_PATH = f"/app/models/{TRANSFORMERS_FILE}"
# ... (rest of the file is the same)

# --- Configuration (MUST match the columns used in your notebook) ---
NUM_COLS = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'step']
NEW_NUM_FEATURES = ['balance_error', 'drained_account']
ALL_NUM_COLS = NUM_COLS + NEW_NUM_FEATURES
CAT_COLS = ['type']

# --- Global Transformer Variables ---
LOADED_MEANS = None
LOADED_STDS = None
LOADED_ENCODER = None

def load_transformers():
    """Loads the preprocessing tools (means, stds, encoder) from disk."""
    global LOADED_MEANS, LOADED_STDS, LOADED_ENCODER
    
    # Check if already loaded
    if all([LOADED_MEANS, LOADED_STDS, LOADED_ENCODER]):
        return True
        
    try:
        # Load the joblib file
        transformers = joblib.load(TRANSFORMERS_PATH)
        LOADED_MEANS = transformers['means']
        LOADED_STDS = transformers['stds']
        LOADED_ENCODER = transformers['encoder']
        return True
    except FileNotFoundError:
        print(f"Error: Preprocessing file not found at {TRANSFORMERS_PATH}. Please check your 'models' folder.")
        return False
    except Exception as e:
        print(f"Error loading transformers: {e}")
        return False

def preprocess_transaction(transaction_data: dict, model_features: list) -> np.ndarray:
    """
    Applies the exact same feature engineering, scaling, and one-hot encoding
    as done in the Kaggle notebook to a single transaction for real-time inference.
    """
    # Initialize transformers if they haven't been loaded by main.py yet
    if not load_transformers():
         raise Exception("Preprocessing tools failed to load or were not found.")
             
    # 1. Convert to DataFrame (FastAPI input is dict, but prep needs DataFrame)
    df = pd.DataFrame([transaction_data])

    # 2. Feature Engineering (Crucial: Must match notebook steps)
    # balance_error = oldbalanceOrg - newbalanceOrig - amount
    df['balance_error'] = df['oldbalanceOrg'] - df['newbalanceOrig'] - df['amount']
    # drained_account = 1 if newbalanceOrig == 0, else 0
    df['drained_account'] = (df['newbalanceOrig'] == 0).astype(int)

    # 3. Scaling (Numerical Columns)
    for col in ALL_NUM_COLS:
        if col in df.columns and LOADED_STDS.get(col, 0) > 0:
            df[col] = (df[col] - LOADED_MEANS[col]) / LOADED_STDS[col]

    # 4. One-Hot Encoding (Categorical Columns: 'type')
    cat_data = df[CAT_COLS]
    # NOTE: We use .transform() here, NOT .fit_transform()
    encoded_data = LOADED_ENCODER.transform(cat_data)
    encoded_df = pd.DataFrame(encoded_data, columns=LOADED_ENCODER.get_feature_names_out())
    
    # Drop original categorical column and others that are not model inputs
    # 'isFraud' is omitted as it's the target, not an input.
    df_processed = df.drop(columns=CAT_COLS + ['nameOrig', 'nameDest', 'isFlaggedFraud'], errors='ignore')
    
    # Merge numerical and encoded features
    df_processed.reset_index(drop=True, inplace=True)
    encoded_df.reset_index(drop=True, inplace=True)
    final_features = pd.concat([df_processed, encoded_df], axis=1)
    
    # 5. Reindex to model's feature order (CRUCIAL for XGBoost deployment!)
    X_processed = final_features.reindex(columns=model_features, fill_value=0)

    # Convert to numpy array for XGBoost and return
    return X_processed.values