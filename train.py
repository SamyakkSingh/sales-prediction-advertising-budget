import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from utils import load_data, prepare_features

OUT = Path('outputs')
MODELS = Path('models')
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

def main():
    df = load_data('Advertising.csv')
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Linear Regression
    lin = LinearRegression()
    lin.fit(X_train_s, y_train)
    y_pred_lin = lin.predict(X_test_s)
    eval_lin = evaluate(y_test, y_pred_lin)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    eval_rf = evaluate(y_test, y_pred_rf)

    # Save models
    joblib.dump(scaler, MODELS / 'scaler.joblib')
    joblib.dump(lin, MODELS / 'linear_model.joblib')
    joblib.dump(rf, MODELS / 'rf_model.joblib')

    # Save report
    with open(OUT / 'train_report.txt', 'w') as f:
        f.write("Model Evaluation\n")
        f.write(f"Linear Regression: {eval_lin}\n")
        f.write(f"Random Forest: {eval_rf}\n")

    print("Training complete. Models saved in models/")

if __name__ == '__main__':
    main()
