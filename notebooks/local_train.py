# local_train.py - Runs the notebook's training and saves artifacts locally.
# NOTE: Ensure this script is executed from the project root (D:\fraud-MlOps)

import numpy as np 
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib
import os
import zipfile # <-- Add this import

# --- Configuration ---
ZIP_FILE_PATH = "data/archive.zip" # <-- Path to zip file
CSV_FILE_NAME = "fraud.csv"      # <-- Name of the file inside the zip
DATA_DIR = "data"                # <-- Directory to extract to
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_FILE_NAME = "xgboost_fraud_model.json"
PREPROC_FILE_NAME = "preprocessing_transformers.joblib"

CHUNK_SIZE = 100000
TARGET_COL = 'isFraud'
CAT_COLS = ['type']
ORG_NUM_COLS = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'step']
ALL_NUM_COLS = ORG_NUM_COLS + ['balance_error', 'drained_account']

# Ensure the models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)
# Ensure data directory exists for extraction
os.makedirs(DATA_DIR, exist_ok=True)
print("--- Starting Local Training Script ---")

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
    exit()
except Exception as e:
    print(f"ERROR during unzip or load: {e}")
    exit()

# --- 2. Stratified Split (Simplified) ---
fraud_indices = df[df[TARGET_COL] == 1].index.tolist()
non_fraud_indices = df[df[TARGET_COL] == 0].index.tolist()
non_fraud_train_idx, non_fraud_test_idx = train_test_split(non_fraud_indices, test_size=0.2, random_state=56)
fraud_train_idx, fraud_test_idx = train_test_split(fraud_indices, test_size=0.2, random_state=56)
train_indices = non_fraud_train_idx + fraud_train_idx
train_df = df.loc[train_indices]

# --- 3. Feature Engineering and Preprocessor Fitting ---
train_df['balance_error'] = train_df['oldbalanceOrg'] - train_df['newbalanceOrig'] - train_df['amount']
train_df['drained_account'] = (train_df['newbalanceOrig'] == 0).astype(int)

sums = {col: train_df[col].sum() for col in ALL_NUM_COLS}
sum_sqs = {col: (train_df[col]**2).sum() for col in ALL_NUM_COLS}
total_rows = len(train_df)

means = {col: sums[col] / total_rows for col in ALL_NUM_COLS}
stds = {
    col: np.sqrt(sum_sqs[col] / total_rows - means[col]**2) 
    if (sum_sqs[col] / total_rows - means[col]**2) > 0 
    else 1 for col in ALL_NUM_COLS
}

all_unique_type_list = sorted(list(train_df['type'].unique()))
encoder = OneHotEncoder(categories=[all_unique_type_list], handle_unknown='ignore', sparse_output=False)
dummy_df = pd.DataFrame({CAT_COLS[0]: all_unique_type_list})
encoder.fit(dummy_df)

fraud_count = train_df[TARGET_COL].sum()
non_fraud_count = total_rows - fraud_count
scale_pos_weight = non_fraud_count / fraud_count
print(f"Scale Pos Weight calculated: {scale_pos_weight:.2f}")

# --- 4. Prepare Training Data for XGBoost (Apply Preprocessing) ---
X_train_final = train_df.copy()
for col in ALL_NUM_COLS:
    if stds[col] > 0:
         X_train_final[col] = (X_train_final[col] - means[col]) / stds[col]

cat_data = X_train_final[CAT_COLS]
encoded_data = encoder.transform(cat_data)
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
X_train_final = X_train_final.drop(columns=CAT_COLS + ['nameOrig', 'nameDest'], errors='ignore')

X_train_final.reset_index(drop=True, inplace=True)
encoded_df.reset_index(drop=True, inplace=True)
X_train_final = pd.concat([X_train_final, encoded_df], axis=1)

y_train = X_train_final[TARGET_COL]
X_train = X_train_final.drop(columns=[TARGET_COL])

# --- 5. Train and Save Model Artifacts ---
model = xgb.XGBClassifier(
    objective='binary:logistic', 
    eval_metric='aucpr', 
    scale_pos_weight=scale_pos_weight, 
    max_depth=7,
    learning_rate=0.1,
)

print("\nStarting XGBoost training...")
model.fit(X_train, y_train)

# Save the trained model to the absolute path
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME)
model.save_model(MODEL_PATH)
print(f"✅ Model saved successfully to: {MODEL_PATH}")

# Save the preprocessing artifacts
PREPROC_PATH = os.path.join(MODEL_DIR, PREPROC_FILE_NAME)
joblib.dump({
    'means': means,
    'stds': stds,
    'encoder': encoder
}, PREPROC_PATH)
print(f"✅ Preprocessing transformers saved successfully to: {PREPROC_PATH}")

print("\n--- Artifacts are ready for Docker Deployment ---")
