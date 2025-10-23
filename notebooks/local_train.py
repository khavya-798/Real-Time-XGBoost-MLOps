# /notebooks/local_train.py
# Runs training using data expected to be in /app/data within Docker.

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Ensure StandardScaler is imported
import joblib
import os
import zipfile # Import zipfile

# --- Configuration (Paths relative to /app in Docker) ---
ZIP_FILE_PATH = "data/archive.zip" # Path to zip file inside container
CSV_FILE_NAME = "Fraud.csv"      # Name of the file inside the zip
DATA_DIR = "data"                # Directory to extract to inside container
MODEL_DIR = "models"             # Directory to save models inside container
MODEL_FILE_NAME = "xgboost_fraud_model.json"
PREPROC_FILE_NAME = "preprocessing_transformers.joblib"

# Model Features/Target Configuration
TARGET_COL = 'isFraud'
CAT_COLS = ['type']
ORG_NUM_COLS = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'step']
NEW_NUM_FEATURES = ['balance_error', 'drained_account']
ALL_NUM_COLS = ORG_NUM_COLS + NEW_NUM_FEATURES
COLS_TO_DROP_FOR_TRAINING = ['nameOrig', 'nameDest', 'isFlaggedFraud', TARGET_COL]

# Ensure the models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)
# Ensure data directory exists for extraction
os.makedirs(DATA_DIR, exist_ok=True)

print("--- Starting Training Script inside Docker ---")

# --- 1. Unzip and Load Data ---
print(f"Attempting to unzip {ZIP_FILE_PATH}...")
try:
    with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
        # Extract the specific CSV file to the data directory
        zip_ref.extract(CSV_FILE_NAME, path=DATA_DIR)
    print(f"Successfully extracted {CSV_FILE_NAME} to {DATA_DIR}/")

    # Define the path to the extracted CSV
    extracted_csv_path = os.path.join(DATA_DIR, CSV_FILE_NAME)

    df = pd.read_csv(extracted_csv_path)
    print(f"Data loaded successfully. Total rows: {len(df)}")
except FileNotFoundError:
    print(f"ERROR: Zip file not found at {ZIP_FILE_PATH} or CSV ({CSV_FILE_NAME}) not inside zip.")
    exit(1) # Exit with error code for Docker build failure
except Exception as e:
    print(f"ERROR during unzip or load: {e}")
    exit(1) # Exit with error code

# --- 2. Feature Engineering ---
print("Performing feature engineering...")
df['balance_error'] = df['oldbalanceOrg'] - df['newbalanceOrig'] - df['amount']
df['drained_account'] = (df['newbalanceOrig'] == 0).astype(int)

# --- 3. Feature Selection & Splitting ---
print("Splitting data...")
if TARGET_COL not in df.columns:
    print(f"ERROR: Target column '{TARGET_COL}' not found.")
    exit(1)

X = df.drop(columns=COLS_TO_DROP_FOR_TRAINING)
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=56, # Consistent random state
    stratify=y
)
print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

# --- 4. Preprocessor Fitting & Data Transformation ---
print("Preprocessing features...")
X_train_cat = X_train[CAT_COLS]
X_train_num = X_train[ALL_NUM_COLS]
X_test_cat = X_test[CAT_COLS]
X_test_num = X_test[ALL_NUM_COLS]

# Fit and transform OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(X_train_cat)
ohe_feature_names = encoder.get_feature_names_out(CAT_COLS)
X_train_encoded = pd.DataFrame(encoder.transform(X_train_cat), columns=ohe_feature_names, index=X_train.index)
X_test_encoded = pd.DataFrame(encoder.transform(X_test_cat), columns=ohe_feature_names, index=X_test.index)

# Fit and transform StandardScaler for numerical features
scaler = StandardScaler()
scaler.fit(X_train_num)
X_train_scaled_num = pd.DataFrame(scaler.transform(X_train_num), columns=ALL_NUM_COLS, index=X_train.index)
X_test_scaled_num = pd.DataFrame(scaler.transform(X_test_num), columns=ALL_NUM_COLS, index=X_test.index)

# Combine processed features
X_train_processed = pd.concat([X_train_scaled_num, X_train_encoded], axis=1)
X_test_processed = pd.concat([X_test_scaled_num, X_test_encoded], axis=1)
feature_names_out = list(X_train_processed.columns) # Save final feature order

# --- 5. Class Weight Calculation ---
print("Calculating class weight...")
count_non_fraud = y_train.value_counts().get(0, 0)
count_fraud = y_train.value_counts().get(1, 0)
if count_fraud == 0:
    print("WARNING: No fraud samples in training set! scale_pos_weight set to 1.")
    scale_pos_weight = 1
else:
    scale_pos_weight = count_non_fraud / count_fraud
print(f"Scale Pos Weight calculated: {scale_pos_weight:.2f}")

# --- 6. Train XGBoost Model ---
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='aucpr',
    scale_pos_weight=scale_pos_weight,
    max_depth=7,
    learning_rate=0.1,
    use_label_encoder=False # Set explicitly
)
print("\nStarting XGBoost training...")
model.fit(X_train_processed, y_train)

# --- 7. Save Model and Preprocessors ---
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME)
model.save_model(MODEL_PATH)
print(f"✅ Model saved successfully to: {MODEL_PATH}")

PREPROC_PATH = os.path.join(MODEL_DIR, PREPROC_FILE_NAME)
joblib.dump({
    'scaler': scaler, # Save the scaler object
    'encoder': encoder, # Save the encoder object
    'feature_names': feature_names_out # Save the feature names/order
}, PREPROC_PATH)
print(f"✅ Preprocessing transformers saved successfully to: {PREPROC_PATH}")

print("\n--- Training Script Finished ---")

