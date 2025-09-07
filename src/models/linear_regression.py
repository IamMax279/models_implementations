from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def linear_regression(X, y, feature_count):
    try:
        if not feature_count:
            raise ValueError('Unspecified feature count!')

        if X.empty or y.empty:
            raise ValueError('Missing features or target!')

        X_mat = np.array(X).reshape(-1, feature_count)
        y_vec = np.array(y).reshape(-1, 1)


    except ValueError as e:
        print(f"An error ocurred: {e}")

bunch = fetch_california_housing()

data, feature_names = bunch.data, bunch.feature_names
target, target_name = bunch.target, bunch.target_names

df = pd.DataFrame(bunch.data, columns=feature_names)
df['MedHouseVal'] = target

features = df[[
    'MedInc',
    'HouseAge',
    'AveRooms',
    'AveBedrms',
    'Population',
    'AveOccup',
    'Latitude',
    'Longitude'
]]

X_train, X_test, y_train, y_test = train_test_split(features, df['MedHouseVal'], test_size=0.25, random_state=42)

# 8 features for this specific dataset
feature_count = 8

linear_regression(X_train, y_train, feature_count)