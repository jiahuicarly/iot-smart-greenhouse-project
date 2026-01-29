import streamlit as st
import pickle
import numpy as np
import time
import os

# Page config
st.set_page_config(
    page_title="Smart Greenhouse Crop Prediction",
    page_icon="🌿",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
body {
    background-color: #f7fbf8;
}
.main {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.05);
}
h1 {
    color: #2e7d32;
}
h3 {
    color: #388e3c;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
.stButton>button:hover {
    background-color: #45a049;
}
.result-box {
    background-color: #e8f5e9;
    padding: 15px;
    border-radius: 10px;
    border-left: 6px solid #2e7d32;
    font-size: 18px;
    color: #1b5e20;
}
</style>
""", unsafe_allow_html=True)

# Load model, scaler, encoder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "rf_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "model", "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "..", "model", "label_encoder.pkl")

model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))
le = pickle.load(open(ENCODER_PATH, "rb"))

# Title
st.markdown("<h1 style='text-align: center;'>🌿 Smart Greenhouse Crop Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>Simulation-Based IoT Data Analytics Demo</p>", unsafe_allow_html=True)
st.divider()

st.info("This demo simulates IoT sensor inputs to support predictive decision-making in greenhouse environments.")

# Input section
st.subheader("🌱 Enter Environmental Parameters")

col1, col2 = st.columns(2)

with col1:
    temp = st.number_input("🌡️ Temperature (°C)", step=1.0)
    rain = st.number_input("🌧️ Rainfall (mm)", step=0.1)
    ph = st.number_input("🧪 Soil pH", step=0.1)

with col2:
    hum = st.number_input("💧 Humidity (kg/kg)", format="%.4f", step=0.0001)
    n = st.number_input("🟢 Nitrogen (N)", step=1)
    p = st.number_input("🟣 Phosphorus (P)", step=1)
    k = st.number_input("🟠 Potassium (K)", step=1)

st.divider()

# Predict
if st.button("🌾 Predict Crop"):
    with st.spinner("🌿 Growing prediction..."):
        time.sleep(1)

    data = np.array([[temp, hum, rain, ph, n, p, k]])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    crop_name = le.inverse_transform(prediction)

    st.markdown(
        f"<div class='result-box'>🌱 Predicted Crop: <b>{crop_name[0]}</b></div>",
        unsafe_allow_html=True
    )

# Footer
st.markdown("---")
st.caption("Simulation-based smart greenhouse prediction system for academic demonstration.")
