import streamlit as st
import numpy as np
import pickle

# =========================
# Load Model
# =========================

model = pickle.load(
    open("models/knn_classifier.pkl", "rb")
)

metrics = pickle.load(
    open("models/metrics.pkl", "rb")
)

# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="KNN Classification App",
    layout="centered"
)

# =========================
# Title
# =========================

st.title("🎯 KNN Classification App")

st.write("Enter feature values below:")

# =========================
# INPUTS
# Example: Iris Dataset
# =========================

feature1 = st.number_input(
    "Feature 1",
    value=5.1
)

feature2 = st.number_input(
    "Feature 2",
    value=3.5
)

feature3 = st.number_input(
    "Feature 3",
    value=1.4
)

feature4 = st.number_input(
    "Feature 4",
    value=0.2
)

# =========================
# Prediction
# =========================

if st.button("Predict Class"):

    input_data = np.array([[
        feature1,
        feature2,
        feature3,
        feature4
    ]])

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Class: {prediction[0]}"
    )
# =========================
# Model Information
# =========================

st.subheader("📊 Model Information")

col1, col2 = st.columns(2)

with col1:

    st.info(
        f"K Value: {model.named_steps['knn'].n_neighbors}"
    )

    st.info(
        f"Weights: {model.named_steps['knn'].weights}"
    )

with col2:

    st.info(
        f"Metric: {model.named_steps['knn'].metric}"
    )

    st.info(
        f"Features Expected: {model.named_steps['knn'].n_features_in_}"
    )
# =========================
# Metrics
# =========================

st.subheader("📈 Model Metrics")

col3, col4 = st.columns(2)

with col3:

    st.metric(
        "Accuracy",
        f"{metrics['Accuracy']:.2f}"
    )

    st.metric(
        "Precision",
        f"{metrics['Precision']:.2f}"
    )

with col4:

    st.metric(
        "Recall",
        f"{metrics['Recall']:.2f}"
    )

    st.metric(
        "F1 Score",
        f"{metrics['F1 Score']:.2f}"
    )

# =========================
# Footer
# =========================

st.caption(
    "Built using Streamlit & Scikit-Learn"
)