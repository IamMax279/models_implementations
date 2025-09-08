from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def linear_regression(X, y, feature_count, observation):
    try:
        if not feature_count:
            raise ValueError('Unspecified feature count!')

        if X.empty or y.empty:
            raise ValueError('Missing features or target!')

        X_mat = np.array(X).reshape(-1, feature_count)
        y_vec = np.array(y).reshape(-1, 1)

        # X_mat.shape[0] - number of rows of the matrix, .shape[2] - number of cols
        # np.ones((X_mat.shape[0], 1)) - create n x 1 matrix of ones
        # np.hstack(..., X_mat) - put the ones matrix with X_mat together (horizontally)
        X_mat_with_bias = np.hstack([np.ones((X_mat.shape[0], 1)), X_mat])

        # Use OLS (Ordinary Least Squares) to compute the most efficient betas
        beta_coeffs = np.linalg.inv(X_mat_with_bias.T @ X_mat_with_bias) \
        @ (X_mat_with_bias.T @ y_vec)

        prediction = beta_coeffs[0][0]

        for i, c in enumerate(beta_coeffs[1:]):
            prediction += c[0] * observation[0][i]
        
        return prediction

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

test = np.array(X_test.head(1))
res = linear_regression(X_train, y_train, feature_count, test)

print(res)