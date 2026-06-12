import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from io import BytesIO

# ==================================
# PAGE CONFIG
# ==================================

st.set_page_config(
    page_title="Emission Analytics Dashboard",
    page_icon="🌍",
    layout="wide"
)

# ==================================
# LOAD MODELS
# ==================================

regressor = joblib.load("xgboost_regressor.pkl")
classifier = joblib.load("xgboost_classifier.pkl")

label_encoder = joblib.load("label_encoder.pkl")
feature_cols = joblib.load("feature_cols.pkl")
target_cols = joblib.load("target_cols.pkl")

# ==================================
# SIDEBAR
# ==================================

st.sidebar.title("📂 Data Input Options")

input_source = st.sidebar.radio(
    "Select input source:",
    [
        "Upload CSV/Excel",
        "Manual Input",
        "Live Sensor (Simulated)"
    ]
)

forecast_horizon = st.sidebar.selectbox(
    "Forecast Horizon (Months)",
    [6, 12, 24]
)

# ==================================
# PREPROCESSING FUNCTION
# ==================================

def preprocess_data(df):

    df.columns = [c.strip() for c in df.columns]

    df["Date"] = pd.to_datetime(df["Date"])

    pivot_df = df.pivot_table(
        index=["Date", "ATTRIBUTE"],
        columns="PARAMETER",
        values="VALUE",
        aggfunc="first"
    ).reset_index()

    pivot_df.rename(
        columns={"ATTRIBUTE": "Attribute"},
        inplace=True
    )

    pivot_df = pivot_df.sort_values("Date")

    pivot_df["Month_Num"] = pivot_df["Date"].dt.month
    pivot_df["Year_Num"] = pivot_df["Date"].dt.year

    targets = [
        "CO2 (ppm)",
        "CH4 (ppm)",
        "VOC (ppm)",
        "SPM (ug/m3)"
    ]

    for col in targets:

        pivot_df[f"{col}_lag1"] = pivot_df[col].shift(1)
        pivot_df[f"{col}_lag2"] = pivot_df[col].shift(2)
        pivot_df[f"{col}_lag3"] = pivot_df[col].shift(3)

    pivot_df = pivot_df.dropna()

    return pivot_df

# ==================================
# TITLE
# ==================================

st.title("🌍 Intelligent Digital Platform for Emissions Prediction")

st.markdown(
    """
    Upload plant environmental data,
    predict emissions,
    classify emission source,
    and forecast future emissions.
    """
)

uploaded_file = None

if input_source == "Upload CSV/Excel":

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel File",
        type=["csv", "xlsx"]
    )
