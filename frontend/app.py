import streamlit as st
import pandas as pd
import requests
import os
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
# The API_URL uses the service name 'backend' in the Docker network
API_URL = os.getenv("API_URL", "http://localhost:8000/predict") 
# The data file path is still needed for loading sample data if we re-enable that logic,
# but for the manual form, this constant ensures clarity.
DATA_PATH = "/app/data/fraud.csv" 

# --- SVG Icons (for branding and social links) ---
# Using Lucide icons for consistency (Streamlit's default)
GITHUB_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-github"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.75c3.25 0 6.8-.75 6.8-7a5.5 5.5 0 0 0-1.5-3.5 5.2 5.2 0 0 0-1 4.75 5.5 5.5 0 0 0-1 4.75"></path><path d="M12 17c-2.43 0-5.46-1.58-6.8-7a5.5 5.5 0 0 0-1.5 3.5c0 6.25 3.55 7 6.8 7a5.2 5.2 0 0 0 1-4.75 5.5 5.5 0 0 0 1 4.75"></path><path d="M12 2v20"></path></svg>'
LINKEDIN_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-linkedin"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path><rect width="4" height="12" x="2" y="9"></rect><circle cx="4" cy="4" r="2"></circle></svg>'
CLOCK_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-clock-1"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l2.5 2.5"/></svg>'
DOLLAR_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-dollar-sign"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6.5"/></svg>'
ACCOUNT_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-user-2"><circle cx="12" cy="4" r="2"/><path d="M12 6c-3.1 0-7 2.1-7 5v2a2 2 0 0 0 2 2h2v4l3-2 3 2v-4h2a2 2 0 0 0 2-2v-2c0-2.9-3.9-5-7-5z"/></svg>'
BANK_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-banknote"><rect width="20" height="12" x="2" y="6" rx="3"/><path d="M12 12m-3 0a3 3 0 1 0 6 0a3 3 0 1 0 -6 0"/><path d="M2 18h20"/><path d="M7 6v2"/></svg>'
TRANSFER_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucede-move-horizontal"><polyline points="18 8 22 12 18 16"/><polyline points="6 8 2 12 6 16"/><line x1="2" y1="12" x2="22" y2="12"/></svg>'


# --- Custom Styling ---
def load_css(file_name):
    """Loads a CSS file and injects it into the app."""
    try:
        # Load the CSS file from the mounted /app/frontend folder
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Using default styling.")

# Load the custom CSS
load_css("frontend/style.css")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Real-Time Fraud Detector",
    page_icon="💳",
    layout="wide"
)

# --- SIDEBAR (Branding and MLOps Info) ---
with st.sidebar:
    st.header("MLOps Architecture")
    st.markdown("""
        This application demonstrates a production-ready MLOps pipeline for **real-time inference**:
        - **Frontend:** Streamlit provides the user interface.
        - **Backend (API):** **FastAPI** serves the model with sub-50ms latency.
        - **Model Logic:** **XGBoost** classifier handles prediction, along with **Joblib-saved preprocessors** for feature engineering and scaling.
        - **Orchestration:** **Docker Compose** manages the networking and deployment of both services.
        """)
    st.markdown("---")
    st.markdown("### Developed By")
    
    # Using markdown and SVG icons for better visual appeal
    st.markdown(f"""
        <div class="profile-link">
            <span>**Khavya**</span> 
            <a href="https://github.com/khavya-798" target="_blank">{GITHUB_SVG} GitHub</a>
            <a href="https://www.linkedin.com/in/khavyanjali-gopisetty-019720254/" target="_blank">{LINKEDIN_SVG} LinkedIn</a>
        </div>
        """, unsafe_allow_html=True)


# --- HEADER AND INFO ---
st.title("💳 Real-Time Fraud Detection Estimator (MLOps)")
st.markdown("### Deployed XGBoost Model for High-Velocity Transaction Screening")
st.markdown("Enter a transaction's raw details below to get an instant fraud probability estimate using the fully containerized pipeline.")

# --- 1. PREDICTION LOGIC ---
def get_prediction(data_dict):
    """Sends a transaction to the FastAPI backend for prediction."""
    try:
        # Call the /predict endpoint
        response = requests.post(API_URL, json=data_dict, timeout=15)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 422:
            st.error(f"API Validation Error: Check Input Data Types/Schema. Response: {response.text}")
            return None
        else:
            st.error(f"API Error: Status {response.status_code}. Response: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Connection Error: Could not connect to the FastAPI backend. Is the 'backend' service running and healthy?")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out (15s limit). The backend service may be slow or overloaded.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during API call: {e}")
        return None

# --- 2. INPUT FORM ---

st.header("1. Enter Transaction Details")
st.markdown('<div class="input-card">', unsafe_allow_html=True)

# Create input fields for all 10 raw features in the required order
with st.form("transaction_form"):
    
    st.info("Note: The model was trained on class-weighted data, ensuring high sensitivity to fraud (Recall).")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Input with Clock Icon
        st.markdown(f"**Time Step (Hour/Day)** {CLOCK_SVG}", unsafe_allow_html=True)
        time_step = st.number_input("", min_value=1, value=50, step=1, key="step")
        
        # Input with Dollar Icon
        st.markdown(f"**Transaction Amount** {DOLLAR_SVG}", unsafe_allow_html=True)
        amount = st.number_input("", min_value=0.01, value=50000.00, step=1000.00, help="The value of the transaction.", key="amount")
        
        # Input with Transfer Icon
        st.markdown(f"**Transaction Type** {TRANSFER_SVG}", unsafe_allow_html=True)
        txn_type = st.selectbox("", 
                                options=['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT'],
                                index=0, key="txn_type")
        
        # Input with Bank Icon
        st.markdown(f"**Originator: Initial Balance (oldbalanceOrg)** {BANK_SVG}", unsafe_allow_html=True)
        oldbalance_org = st.number_input("", min_value=0.0, value=100000.00, step=1000.00, key="oldbalance_org")
        
        # Input with Bank Icon
        st.markdown(f"**Originator: New Balance (newbalanceOrig)** {BANK_SVG}", unsafe_allow_html=True)
        newbalance_org = st.number_input("", min_value=0.0, value=50000.00, step=1000.00, key="newbalance_org")

    with col2:
        # Input with Account Icon
        st.markdown(f"**Originator ID (Cxxxx)** {ACCOUNT_SVG}", unsafe_allow_html=True)
        name_orig = st.text_input("", value="C1234567", help="Unique ID of the paying customer.", key="name_orig")
        
        # Input with Account Icon
        st.markdown(f"**Destination ID (Mxxxx or Cxxxx)** {ACCOUNT_SVG}", unsafe_allow_html=True)
        name_dest = st.text_input("", value="M9876543", help="Unique ID of the recipient.", key="name_dest")
        
        # Input with Bank Icon
        st.markdown(f"**Destination: Initial Balance (oldbalanceDest)** {BANK_SVG}", unsafe_allow_html=True)
        oldbalance_dest = st.number_input("", min_value=0.0, value=20000.00, step=1000.00, key="oldbalance_dest")
        
        # Input with Bank Icon
        st.markdown(f"**Destination: New Balance (newbalanceDest)** {BANK_SVG}", unsafe_allow_html=True)
        newbalance_dest = st.number_input("", min_value=0.0, value=70000.00, step=1000.00, key="newbalance_dest")
        
        # Input for Flagged Fraud
        is_flagged_fraud = st.selectbox("Is Flagged Fraud (Raw System)", options=[0, 1], index=0, help="Flag set by the simple system (0=No, 1=Yes).", key="is_flagged_fraud")

    # Every form needs a submit button
    submitted = st.form_submit_button("Predict Fraud Status (Run MLOps Pipeline)", type="primary")

st.markdown('</div>', unsafe_allow_html=True)

# --- 3. PREDICTION OUTPUT ---

st.header("2. MLOps Prediction Result")

if submitted:
    
    # 1. Collect all data into a dictionary (matching the FastAPI schema)
    transaction_data = {
        "step": time_step,
        "type": txn_type,
        "amount": amount,
        "nameOrig": name_orig,
        "oldbalanceOrg": oldbalance_org,
        "newbalanceOrig": newbalance_org,
        "nameDest": name_dest,
        "oldbalanceDest": oldbalance_dest,
        "newbalanceDest": newbalance_dest,
        "isFlaggedFraud": is_flagged_fraud
    }
    
    # 2. Call the backend service
    with st.spinner("Processing data, running feature engineering, scaling, and inference on FastAPI..."):
        prediction_result = get_prediction(transaction_data)

    st.markdown("---")
    
    if prediction_result:
        # Check if the model loaded flag is still false, indicating a backend failure
        if prediction_result.get('error'):
            st.error(f"❌ Backend Initialization Failed: {prediction_result['error']}")
            st.warning("Please check your Docker Compose logs for the specific file loading error (e.g., missing model files in the 'models' directory).")
        else:
            is_fraud = prediction_result.get('is_fraud', False)
            prob = prediction_result.get('probability_fraud', 0) * 100
            
            # 3. Display the result
            if is_fraud:
                st.error(f"🚨 **FRAUD DETECTED!** - IMMEDIATE BLOCK SUGGESTED")
            else:
                st.success(f"✅ **Transaction is Likely Genuine.**")
                
            st.metric(
                label="Probability of Fraud (Model Confidence)", 
                value=f"{prob:.4f}%"
            )
            
            with st.expander("Show Raw API Response"):
                st.json(prediction_result)
