from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model(weights, X, y_test, feature_name):
    X_test_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])

    y_pred = (X_test_with_bias @ weights).flatten()

    plt.figure(figsize=(10, 6))
    plt.title('Actual data vs predictions')

    sns.scatterplot(x=X[feature_name], y=y_test, color='blue')
    sns.scatterplot(x=X[feature_name], y=y_pred, color='red')
    plt.xlabel(feature_name)
    plt.ylabel('MedHouseVal')

    plt.show()

def train_linear_regression(X, y, feature_count):
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
        weights = np.linalg.inv(X_mat_with_bias.T @ X_mat_with_bias) \
        @ (X_mat_with_bias.T @ y_vec)

        return weights

    except ValueError as e:
        print(f"An error ocurred: {e}")

def predict(weights, X):
    X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    return X_with_bias @ weights

def get_score(y, predictions):
    y_test_vec = np.array(y).reshape(-1, 1)
    rss = sum((y_test_vec - predictions)**2)
    tss = sum((y_test_vec - np.mean(y_test_vec))**2)
    r_sq = 1 - (rss / tss)
    return r_sq


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

weights = train_linear_regression(X_train, y_train, feature_count, X_test)
predictions = predict(weights, X_test)
score = get_score(y_test, predictions)

plot_model(weights, X_test, y_test, 'AveRooms')