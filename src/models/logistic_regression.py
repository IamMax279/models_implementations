import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, feature_count, lr=0.01, iterations=1000):
    X_mat = np.array(X).reshape(-1, feature_count)
    y_vec = np.array(y).reshape(-1, 1)

    # np.ones, np.zeros etc. expect a single argument - a tuple with matrix size
    X_with_bias = np.hstack([np.ones((X_mat.shape[0], 1)), X_mat])

    weights = np.zeros((X_with_bias.shape[1], 1))

    # observation count
    m = X_with_bias.shape[0]

    for i in range(iterations):
        # compute predictions for each observation
        z = X_with_bias @ weights
        epsilon = 1e-15
        h = np.clip(sigmoid(z), epsilon, 1 - epsilon)
        gradient = 1/m * (X_with_bias.T @ (h - y_vec))
        weights -= lr * gradient
        if i % 100 == 0:
            # log-loss
            cost = -(1/m) * np.sum(y_vec * np.log(h) + (1-y_vec) * np.log(1-h))
            # show the cost variable with floating point precision of 4
            print(f"Iteration {i}: cost = {cost:.4f}")

    print(f"weights: {weights}")

df = pd.read_csv('src/data/diabetes.csv', sep=',')

map_genders = {
    'Male': 1,
    'Female': 0
}

df['gender'] = df['gender'].map(map_genders)

map_smoking_history = {
    'No Info': 0,
    'never': 1,
    'former': 2,
    'not current': 3,
    'ever': 4,
    'current': 5
}

df['smoking_history'] = df['smoking_history'].map(map_smoking_history)

features = df[[
    'gender',
    'age',
    'hypertension',
    'heart_disease',
    'smoking_history',
    'bmi',
    'HbA1c_level',
    'blood_glucose_level'
]]

X_train, X_test, y_train, y_test = train_test_split(features, df['diabetes'], test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_standardized = scaler.fit_transform(X_train)

feature_count = len(np.array(features)[0])
sample = [1, 29.0, 1, 0, 1, 33.76, 5.9, 119]
logistic_regression(X_train_standardized, y_train, feature_count)