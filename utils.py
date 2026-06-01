import pandas as pd
import numpy as np

def load_data(path='Advertising.csv'):
    df = pd.read_csv(path)
    return df

def prepare_features(df):
    X = df[['TV', 'Radio', 'Newspaper']].values
    y = df['Sales'].values
    return X, y
