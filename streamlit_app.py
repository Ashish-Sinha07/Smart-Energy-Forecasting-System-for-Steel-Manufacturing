import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# Load Models
# ===============================
@st.cache_resource
def load_models():
    reg_model = joblib.load("regression_pipeline.pkl")
    cls_model = joblib.load("classification_pipeline.pkl")
    label_enc = joblib.load("label_encoder.pkl")
    return reg_model, cls_model, label_enc

reg_pipeline, cls_pipeline, label_enc = load_models()

# ===============================
# App UI
# ===============================
st.set_page_config(page_title="Steel Industry Load Forecasting", layout="wide")

st.title("⚙️ Steel Industry Energy Forecasting Dashboard")
st.write("Predict plant energy usage and load type using ML models")

# ===============================
# User Inputs
# ===============================
st.header("🔧 Input Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    lag_kvarh = st.number_input("Lagging Reactive Power (kVarh)", min_value=0.0, value=200.0)
    leading_kvarh = st.number_input("Leading Reactive Power (kVarh)", min_value=0.0, value=100.0)
    co2 = st.number_input("CO2 Emission (tCO2)", min_value=0.0, value=0.5)

with col2:
    lag_pf = st.number_input("Lagging Power Factor", min_value=0.0, max_value=1.0, value=0.85)
    lead_pf = st.number_input("Leading Power Factor", min_value=0.0, max_value=1.0, value=0.10)
    nsm = st.number_input("NSM (Minutes in Day)", min_value=0, value=780)

with col3:
    hour = st.number_input("Hour of Day (0-23)", min_value=0, max_value=23, value=14)
    day = st.number_input("Day of Month (1-31)", min_value=1, max_value=31, value=10)
    month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=5)

week = st.number_input("Week Number (1-52)", min_value=1, max_value=52, value=20)

day_of_week = st.selectbox(
    "Day of Week",
    ["0", "1", "2", "3", "4", "5", "6"]  # If originally numeric
)

week_status = st.selectbox(
    "Week Status",
    ["Weekend", "Weekday"]
)

# ===============================
# Feature Engineering (same as training)
# ===============================
is_weekend = 1 if day_of_week in ["5", "6"] else 0
is_peak_hour = 1 if 8 <= hour <= 20 else 0

total_reactive = lag_kvarh + leading_kvarh
net_reactive = lag_kvarh - leading_kvarh
pf_diff = lag_pf - lead_pf

energy_intensity = 0
if nsm != 0:
    energy_intensity = lag_kvarh / nsm

# Create dataframe for model
input_data = pd.DataFrame([{
    'Lagging_Current_Reactive.Power_kVarh': lag_kvarh,
    'Leading_Current_Reactive_Power_kVarh': leading_kvarh,
    'CO2(tCO2)': co2,
    'Lagging_Current_Power_Factor': lag_pf,
    'Leading_Current_Power_Factor': lead_pf,
    'NSM': nsm,
    'Hour': hour,
    'Day': day,
    'Month': month,
    'Week': week,
    'Is_Weekend': is_weekend,
    'Is_Peak_Hour': is_peak_hour,
    'Total_Reactive_Power_kVarh': total_reactive,
    'Net_Reactive_Power': net_reactive,
    'PF_Difference': pf_diff,
    'Energy_Intensity': energy_intensity,
    'Day_of_week_str': day_of_week,
    'WeekStatus': week_status
}])

# ===============================
# Predict
# ===============================
if st.button("Predict"):

    # Regression prediction
    pred_usage = reg_pipeline.predict(input_data)[0]

    # Classification prediction (encoded to original class)
    cls_pred_enc = cls_pipeline.predict(input_data)[0]
    cls_pred = label_enc.inverse_transform([cls_pred_enc])[0]

    st.subheader("📌 Prediction Results")
    st.write(f"### 🔋 Predicted Usage (kWh): `{pred_usage:.2f}`")
    st.write(f"### 📦 Predicted Load Type: `{cls_pred}`")