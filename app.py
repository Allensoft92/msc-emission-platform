import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODELS
# =========================

regressor = joblib.load("xgboost_regressor.pkl")
classifier = joblib.load("xgboost_classifier.pkl")

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Emission Prediction Platform",
    layout="wide"
)

st.title("Intelligent Digital Platform for Emissions Prediction")

st.markdown("""
This platform predicts:

- CO₂
- CH₄
- VOC
- SPM

and identifies likely emission sources:

- FA (Flare Area)
- CA (Compressor Area)
- OA (Office Area)
- TA (Turbine Area)

Powered by XGBoost.
""")

# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df)

    st.success("File uploaded successfully.")

else:

    st.info("Upload a CSV file to continue.")