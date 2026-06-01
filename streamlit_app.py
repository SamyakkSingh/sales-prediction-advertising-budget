import streamlit as st
import numpy as np
import joblib
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Sales Prediction App", layout="centered")

MODELS = Path("models")

def load_models():
    scaler = joblib.load(MODELS / "scaler.joblib")
    lin = joblib.load(MODELS / "linear_model.joblib")
    rf = joblib.load(MODELS / "rf_model.joblib")
    return scaler, lin, rf

st.title("Sales Prediction App")
st.write("Predict sales using advertising budgets (TV, Radio, Newspaper).")

if Path("Advertising.csv").exists():
    if st.checkbox("Show Data Preview(displays the dataset)"):
        df = pd.read_csv("Advertising.csv")
        st.dataframe(df.head())

tv = st.number_input("TV Budget(in thousands of dollars)", min_value=0.0, max_value=1000.0, value=150.0)
radio = st.number_input("Radio Budget(in thousands of dollars)", min_value=0.0, max_value=1000.0, value=20.0)
newspaper = st.number_input("Newspaper Budget(in thousands of dollars)", min_value=0.0, max_value=1000.0, value=10.0)

if st.button("Predict"):
    try:
        scaler, lin, rf = load_models()
        X = np.array([[tv, radio, newspaper]])
        Xs = scaler.transform(X)

        pred_lin = lin.predict(Xs)[0]
        pred_rf = rf.predict(X)[0]

        st.success(f"Linear Regression Predicted Sales: {pred_lin:.3f}")
        st.success(f"Random Forest Predicted Sales: {pred_rf:.3f}")

    except:
        st.error("Model files not found. Please run train.py first!")

st.markdown("---")
st.caption("Run with: streamlit run streamlit_app.py")
