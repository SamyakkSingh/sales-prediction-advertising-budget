import argparse
import joblib
import numpy as np
from pathlib import Path

MODELS = Path('models')

def load_models():
    scaler = joblib.load(MODELS / 'scaler.joblib')
    lin = joblib.load(MODELS / 'linear_model.joblib')
    rf = joblib.load(MODELS / 'rf_model.joblib')
    return scaler, lin, rf

def predict_sales(tv, radio, newspaper):
    scaler, lin, rf = load_models()
    X = np.array([[tv, radio, newspaper]])
    Xs = scaler.transform(X)
    pred_lin = lin.predict(Xs)[0]
    pred_rf = rf.predict(X)[0]
    return pred_lin, pred_rf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tv", type=float, required=True)
    parser.add_argument("--radio", type=float, required=True)
    parser.add_argument("--newspaper", type=float, required=True)
    args = parser.parse_args()

    lin, rf = predict_sales(args.tv, args.radio, args.newspaper)
    print("Predicted Sales (Linear Regression):", lin)
    print("Predicted Sales (Random Forest):", rf)
