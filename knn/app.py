import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================
# Load Model
# =========================

model = pickle.load(
    open("models/knn_model.pkl", "rb")
)

# Optional Metrics File
try:
    metrics = pickle.load(
        open("models/metrics.pkl", "rb")
    )
except:
    metrics = None

# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="KNN House Price Prediction",
    layout="centered"
)

# =========================
# Title
# =========================

st.title("🏠 KNN House Price Prediction App")

st.write(
    "Enter the house details below:"
)

# =========================
# Input Fields
# =========================

transaction_date = st.number_input(
    "Transaction Date",
    value=2013.0
)

house_age = st.number_input(
    "House Age",
    value=10.0
)

distance_to_mrt = st.number_input(
    "Distance to MRT Station",
    value=300.0
)

num_convenience_stores = st.number_input(
    "Number of Convenience Stores",
    value=5
)

latitude = st.number_input(
    "Latitude",
    value=24.97
)

longitude = st.number_input(
    "Longitude",
    value=121.54
)

# =========================
# Prediction
# =========================


# =========================
# Model Information
# =========================

st.subheader("📊 Model Information")

col1, col2 = st.columns(2)

with col1:

    st.info(
        f"""
        Algorithm:
        KNeighborsRegressor
        """
    )

    st.info(
        f"""
        K Value:
        {model.n_neighbors}
        """
    )

with col2:

    st.info(
        f"""
        Metric:
        {model.metric}
        """
    )

    st.info(
        f"""
        Weights:
        {model.weights}
        """
    )

# =========================
# Metrics
# =========================

st.subheader("📈 Model Metrics")

if metrics:

    col3, col4 = st.columns(2)

    with col3:

        st.metric(
            "MAE",
            f"{metrics['MAE']:.2f}"
        )

        st.metric(
            "RMSE",
            f"{metrics['RMSE']:.2f}"
        )

    with col4:

        st.metric(
            "MSE",
            f"{metrics['MSE']:.2f}"
        )

        st.metric(
            "R2 Score",
            f"{metrics['R2']:.2f}"
        )

else:

    st.warning(
        "Metrics file not found."
    )

if st.button("Predict House Price"):

    input_data = np.array([[
        transaction_date,
        house_age,
        distance_to_mrt,
        num_convenience_stores,
        latitude,
        longitude
    ]])

    prediction = model.predict(input_data)

    st.success(
        f"Predicted House Price Per Unit Area: {prediction[0]:.2f}"
    )

# =========================
# Footer
# =========================

st.caption(
    "Built with Streamlit & Scikit-Learn"
)