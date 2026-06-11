import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Emission Prediction Platform",
    page_icon="🌍",
    layout="wide"
)

# =========================
# LOAD FILES
# =========================

regressor = joblib.load("xgboost_regressor.pkl")
classifier = joblib.load("xgboost_classifier.pkl")

label_encoder = joblib.load("label_encoder.pkl")
feature_cols = joblib.load("feature_cols.pkl")
target_cols = joblib.load("target_cols.pkl")

# =========================
# SIDEBAR
# =========================

page = st.sidebar.selectbox(
    "Navigation",
    [
        "Home",
        "Model Information",
        "Emission Prediction",
        "Source Identification"
    ]
)

# =========================
# HOME
# =========================

if page == "Home":

    st.title("🌍 Intelligent Digital Platform for Emissions Prediction")

    st.markdown("""
    ### MSc Research Project

    Development of an Intelligent Digital Platform for Emissions Prediction
    in Gas Processing Facilities Using Machine Learning.

    ### Predicted Parameters

    - CO₂
    - CH₄
    - VOC
    - SPM

    ### Source Categories

    - CA (Compressor Area)
    - FA (Flare Area)
    - OA (Office Area)
    - TA (Turbine Area)

    ### Machine Learning Model

    XGBoost Regressor and XGBoost Classifier
    """)

# =========================
# MODEL INFO
# =========================

elif page == "Model Information":

    st.header("Model Information")

    st.write("Regression Model: XGBoost Regressor")

    st.write("Classification Model: XGBoost Classifier")

    st.write("Feature Count:", len(feature_cols))

    st.write("Targets:")

    st.write(target_cols)

# =========================
# EMISSION PREDICTION
# =========================

elif page == "Emission Prediction":

    st.header("Emission Prediction")

    month_num = st.number_input(
        "Month Number",
        min_value=1,
        max_value=12,
        value=1
    )

    year_num = st.number_input(
        "Year Number",
        min_value=2021,
        max_value=2100,
        value=2025
    )

    co2_lag1 = st.number_input("CO2 Lag 1")
    co2_lag2 = st.number_input("CO2 Lag 2")
    co2_lag3 = st.number_input("CO2 Lag 3")

    ch4_lag1 = st.number_input("CH4 Lag 1")
    ch4_lag2 = st.number_input("CH4 Lag 2")
    ch4_lag3 = st.number_input("CH4 Lag 3")

    voc_lag1 = st.number_input("VOC Lag 1")
    voc_lag2 = st.number_input("VOC Lag 2")
    voc_lag3 = st.number_input("VOC Lag 3")

    spm_lag1 = st.number_input("SPM Lag 1")
    spm_lag2 = st.number_input("SPM Lag 2")
    spm_lag3 = st.number_input("SPM Lag 3")

    if st.button("Predict Emissions"):

        X = pd.DataFrame(
            [[
                month_num,
                year_num,
                co2_lag1,
                co2_lag2,
                co2_lag3,
                ch4_lag1,
                ch4_lag2,
                ch4_lag3,
                voc_lag1,
                voc_lag2,
                voc_lag3,
                spm_lag1,
                spm_lag2,
                spm_lag3
            ]],
            columns=feature_cols
        )

        prediction = regressor.predict(X)

        st.success("Prediction Completed")

        result = pd.DataFrame(
            prediction,
            columns=target_cols
        )

        st.dataframe(result)

# =========================
# CLASSIFICATION
# =========================

elif page == "Source Identification":

    st.header("Source Identification")

    month_num = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=1,
        key="month"
    )

    year_num = st.number_input(
        "Year",
        min_value=2021,
        max_value=2100,
        value=2025,
        key="year"
    )

    co2_lag1 = st.number_input("CO2 Lag1", key="co2_1")
    co2_lag2 = st.number_input("CO2 Lag2", key="co2_2")
    co2_lag3 = st.number_input("CO2 Lag3", key="co2_3")

    ch4_lag1 = st.number_input("CH4 Lag1", key="ch4_1")
    ch4_lag2 = st.number_input("CH4 Lag2", key="ch4_2")
    ch4_lag3 = st.number_input("CH4 Lag3", key="ch4_3")

    voc_lag1 = st.number_input("VOC Lag1", key="voc_1")
    voc_lag2 = st.number_input("VOC Lag2", key="voc_2")
    voc_lag3 = st.number_input("VOC Lag3", key="voc_3")

    spm_lag1 = st.number_input("SPM Lag1", key="spm_1")
    spm_lag2 = st.number_input("SPM Lag2", key="spm_2")
    spm_lag3 = st.number_input("SPM Lag3", key="spm_3")

    if st.button("Identify Source"):

        X = pd.DataFrame(
            [[
                month_num,
                year_num,
                co2_lag1,
                co2_lag2,
                co2_lag3,
                ch4_lag1,
                ch4_lag2,
                ch4_lag3,
                voc_lag1,
                voc_lag2,
                voc_lag3,
                spm_lag1,
                spm_lag2,
                spm_lag3
            ]],
            columns=feature_cols
        )

        pred = classifier.predict(X)

        area = label_encoder.inverse_transform(pred)

        st.success(
            f"Predicted Source: {area[0]}"
        )
