# Sales Prediction Using Machine Learning

## Overview

This project predicts product sales based on advertising budgets allocated to TV, Radio, and Newspaper marketing campaigns. It uses Machine Learning regression models to estimate sales and provides predictions through both a command-line interface and an interactive Streamlit web application.

The project compares the performance of Linear Regression and Random Forest Regression models to generate accurate sales predictions from advertising expenditure data.

---

## Features

* Predict sales using advertising budgets for TV, Radio, and Newspaper.
* Train and evaluate multiple Machine Learning models.
* Compare Linear Regression and Random Forest Regression performance.
* Interactive Streamlit web application for easy predictions.
* Command-line prediction support.
* Automatic model saving using Joblib.
* Generates evaluation reports after training.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib
* Streamlit

---

## Dataset

The project uses the **Advertising Dataset**, which contains advertising spending information across different media channels along with corresponding sales figures.

### Input Features

* TV Advertising Budget
* Radio Advertising Budget
* Newspaper Advertising Budget

### Target Variable

* Sales

---

## Project Structure

```text
Sales_Prediction/
│
├── Advertising.csv
├── train.py
├── predict.py
├── streamlit_app.py
├── utils.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── scaler.joblib
│   ├── linear_model.joblib
│   └── rf_model.joblib
│
├── outputs/
│   └── train_report.txt
│
└── __pycache__/
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/SamyakkSingh/sales-prediction-advertising-budget
cd sales-prediction-advertising-budget
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Model Training

Train the Machine Learning models using:

```bash
python train.py
```

This script:

* Loads and preprocesses the dataset.
* Splits data into training and testing sets.
* Trains Linear Regression and Random Forest Regression models.
* Evaluates model performance.
* Saves trained models in the `models/` directory.
* Generates a training report in the `outputs/` directory.

---

## Making Predictions from Command Line

Use the following command:

```bash
python predict.py --tv 150 --radio 20 --newspaper 10
```

Example Output:

```text
Predicted Sales (Linear Regression): 16.2
Predicted Sales (Random Forest): 15.9
```

---

## Running the Streamlit Application

Start the Streamlit web application:

```bash
streamlit run streamlit_app.py
```

The application allows users to:

* Enter advertising budgets.
* Generate sales predictions instantly.
* Compare predictions from multiple models.
* View dataset samples.

---

## Models Used

### Linear Regression

A simple regression algorithm that models the relationship between advertising budgets and sales using a linear equation.

### Random Forest Regression

An ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

---

## Evaluation Metrics

The following metrics are used to evaluate model performance:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

These metrics are automatically generated and saved in the training report.

---

## Future Improvements

* Hyperparameter tuning using GridSearchCV.
* Additional regression models such as XGBoost and Gradient Boosting.
* Advanced visualizations and dashboards.
* Model deployment using cloud platforms.
* Real-time advertising budget optimization.

---

## Author

Developed as a Machine Learning project using Python, Scikit-learn, and Streamlit.
