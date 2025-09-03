# fraud_dashboard.py
import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("fraud_xgb.pkl")

# Page setup
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

# Header
st.markdown(
    """
    <h1 style='text-align: center; color: #FF4B4B;'>Credit Card Fraud Detection</h1>
    <p style='text-align: center; color: gray;'>Enter transaction details in the left side menu bar and then click on the "Predict Fraud" button in the main menu to predict if it's Fraudulent or Not Fraudulent</p>
    """,
    unsafe_allow_html=True
)

# Sidebar for inputs
st.sidebar.header("Input Transaction Details")

distance_from_home = st.sidebar.number_input("Distance from Home", min_value=0.0, value=10.0, step=1.0)
distance_from_last_transaction = st.sidebar.number_input("Distance from Last Transaction", min_value=0.0, value=5.0, step=1.0)
ratio_to_median_purchase_price = st.sidebar.number_input("Ratio to Median Purchase Price", min_value=0.0, value=1.0, step=0.1)

repeat_retailer = st.sidebar.selectbox("Repeat Retailer?", [0, 1])
used_chip = st.sidebar.selectbox("Used Chip?", [0, 1])
used_pin_number = st.sidebar.selectbox("Used PIN Number?", [0, 1])
online_order = st.sidebar.selectbox("Online Order?", [0, 1])

threshold = st.sidebar.slider("Fraud Probability Threshold", 0.0, 1.0, 0.50, 0.01)

# Main section
st.markdown("### Prediction Results")

if st.button("Predict Fraud"):
    features = np.array([[
        distance_from_home,
        distance_from_last_transaction,
        ratio_to_median_purchase_price,
        repeat_retailer,
        used_chip,
        used_pin_number,
        online_order
    ]])

    prob_fraud = float(model.predict_proba(features)[0][1])
    pred = int(prob_fraud >= threshold)

    st.markdown(f"**Fraud Probability:** `{prob_fraud:.4f}`")

    if pred == 1:
        st.error("Fraudulent Transaction Detected!")
    else:
        st.success("Legitimate Transaction")

    # Gauge-like feedback
    st.progress(min(prob_fraud, 1.0))

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: gray;'>
    Built using Streamlit | XGBoost Model
    </p>
    """,
    unsafe_allow_html=True
)
