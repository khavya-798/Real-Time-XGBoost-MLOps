import streamlit as st
import pandas as pd
import requests
import os
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
# Use BACKEND_URL consistently, construct API_URL from it
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_URL = f"{BACKEND_URL}/predict"

# --- SVG Icons (for branding and social links) ---
GITHUB_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-github"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.75c3.25 0 6.8-.75 6.8-7a5.5 5.5 0 0 0-1.5-3.5 5.2 5.2 0 0 0-1 4.75 5.5 5.5 0 0 0-1 4.75"></path><path d="M12 17c-2.43 0-5.46-1.58-6.8-7a5.5 5.5 0 0 0-1.5 3.5c0 6.25 3.55 7 6.8 7a5.2 5.2 0 0 0 1-4.75 5.5 5.5 0 0 0 1 4.75"></path><path d="M12 2v20"></path></svg>'
LINKEDIN_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-linkedin"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path><rect width="4" height="12" x="2" y="9"></rect><circle cx="4" cy="4" r="2"></circle></svg>'
CLOCK_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-clock-1"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l2.5 2.5"/></svg>'
DOLLAR_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-dollar-sign"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6.5"/></svg>'
# ACCOUNT_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-user-2"><circle cx="12" cy="4" r="2"/><path d="M12 6c-3.1 0-7 2.1-7 5v2a2 2 0 0 0 2 2h2v4l3-2 3 2v-4h2a2 2 0 0 0 2-2v-2c0-2.9-3.9-5-7-5z"/></svg>' # Removed as nameOrig/Dest removed
BANK_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-banknote"><rect width="20" height="12" x="2" y="6" rx="3"/><path d="M12 12m-3 0a3 3 0 1 0 6 0a3 3 0 1 0 -6 0"/><path d="M2 18h20"/><path d="M7 6v2"/></svg>'
TRANSFER_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucede-move-horizontal"><polyline points="18 8 22 12 18 16"/><polyline points="6 8 2 12 6 16"/><line x1="2" y1="12" x2="22" y2="12"/></svg>'


# --- Custom Styling ---
def load_css(file_name):
    """Loads a CSS file and injects it into the app."""
    try:
        # Load the CSS file from the /app folder inside container
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Using default styling.")

# Load the custom CSS (assuming it's copied to /app/style.css in Dockerfile)
# load_css("style.css") # Uncomment if you have a style.css file

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
            <span>Khavya</span>
            <a href="https://github.com/khavya-798" target="_blank">{GITHUB_SVG} GitHub</a>
            <a href="https://www.linkedin.com/in/khavyanjali-gopisetty-019720254/" target="_blank">{LINKEDIN_SVG} LinkedIn</a>
        </div>
        """, unsafe_allow_html=True)


# --- HEADER AND INFO ---
st.title("💳 Real-Time Fraud Detection Estimator (MLOps)")
st.markdown("### Deployed XGBoost Model for High-Velocity Transaction Screening")
st.markdown("Enter transaction details below to get an instant fraud probability estimate using the fully containerized pipeline.")

# --- 1. PREDICTION LOGIC ---
def get_prediction(data_dict):
    """Sends a transaction to the FastAPI backend for prediction."""
    try:
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
        st.error(f"Connection Error: Could not connect to the backend API at {BACKEND_URL}. Is the 'backend' service running?")
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

# Create input fields matching the FastAPI Pydantic model
with st.form("transaction_form"):

    st.info("Note: The model was trained on class-weighted data, ensuring high sensitivity to fraud (Recall).")

    col1, col2 = st.columns(2)

    with col1:
        # Input with Clock Icon
        st.markdown(f"**Time Step (Hour/Day)** {CLOCK_SVG}", unsafe_allow_html=True)
        time_step = st.number_input("Step Label", label_visibility="collapsed", min_value=1, value=50, step=1, key="step") # Added label, hidden

        # Input with Dollar Icon
        st.markdown(f"**Transaction Amount** {DOLLAR_SVG}", unsafe_allow_html=True)
        amount = st.number_input("Amount Label", label_visibility="collapsed", min_value=0.01, value=50000.00, format="%.2f", step=1000.00, help="The value of the transaction.", key="amount") # Added label, hidden

        # Input with Transfer Icon
        st.markdown(f"**Transaction Type** {TRANSFER_SVG}", unsafe_allow_html=True)
        txn_type = st.selectbox("Type Label", label_visibility="collapsed",
                                options=['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT'],
                                index=0, key="txn_type") # Added label, hidden

        # Input with Bank Icon
        st.markdown(f"**Originator: Initial Balance** {BANK_SVG}", unsafe_allow_html=True)
        oldbalance_org = st.number_input("OldBalOrg Label", label_visibility="collapsed", min_value=0.0, value=100000.00, format="%.2f", step=1000.00, key="oldbalance_org") # Added label, hidden

    with col2:
        # Input with Bank Icon
        st.markdown(f"**Originator: New Balance** {BANK_SVG}", unsafe_allow_html=True)
        newbalance_org = st.number_input("NewBalOrg Label", label_visibility="collapsed", min_value=0.0, value=50000.00, format="%.2f", step=1000.00, key="newbalance_org") # Added label, hidden

        # Input with Bank Icon
        st.markdown(f"**Destination: Initial Balance** {BANK_SVG}", unsafe_allow_html=True)
        oldbalance_dest = st.number_input("OldBalDest Label", label_visibility="collapsed", min_value=0.0, value=20000.00, format="%.2f", step=1000.00, key="oldbalance_dest") # Added label, hidden

        # Input with Bank Icon
        st.markdown(f"**Destination: New Balance** {BANK_SVG}", unsafe_allow_html=True)
        newbalance_dest = st.number_input("NewBalDest Label", label_visibility="collapsed", min_value=0.0, value=70000.00, format="%.2f", step=1000.00, key="newbalance_dest") # Added label, hidden

        # Removed nameOrig, nameDest, isFlaggedFraud inputs

    # Every form needs a submit button
    submitted = st.form_submit_button("Predict Fraud Status (Run MLOps Pipeline)", type="primary")

st.markdown('</div>', unsafe_allow_html=True)

# --- 3. PREDICTION OUTPUT ---

st.header("2. MLOps Prediction Result")

if submitted:

    # 1. Collect required data into a dictionary (matching the FastAPI schema)
    transaction_data = {
        "step": time_step,
        "type": txn_type,
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "newbalanceOrig": newbalance_org,
        "oldbalanceDest": oldbalance_dest,
        "newbalanceDest": newbalance_dest
    }

    # 2. Call the backend service
    with st.spinner("Processing data, running feature engineering, scaling, and inference on FastAPI..."):
        prediction_result = get_prediction(transaction_data)

    st.markdown("---")

    if prediction_result:
        is_fraud = prediction_result.get('is_fraud', None) # Use None default
        prob_api = prediction_result.get('probability', None) # Correct key, use None default

        # 3. Display the result
        if is_fraud == 1:
            st.error(f"🚨 **FRAUD DETECTED!** - IMMEDIATE BLOCK SUGGESTED")
        elif is_fraud == 0:
            st.success(f"✅ **Transaction is Likely Genuine.**")
        else: # Handle None or unexpected value
             st.warning("⚠️ Prediction could not be determined.")

        if prob_api is not None:
             prob_percent = prob_api * 100 # Convert probability to percentage
             st.metric(
                 label="Probability of Fraud (Model Confidence)",
                 value=f"{prob_percent:.4f}%"
             )
        else:
            st.caption("Fraud probability score unavailable.")

        with st.expander("Show Raw API Response"):
            st.json(prediction_result)
    else:
        # Error message is already shown by get_prediction function
        st.info("Prediction failed. See error message above.")