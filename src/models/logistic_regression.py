import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, feature_count, lr=0.01, iterations=1000):
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
        # (h - y_vec) is the deviation vector
        gradient = 1/m * (X_with_bias.T @ (h - y_vec))
        weights -= lr * gradient
        if i % 100 == 0:
            # log-loss
            cost = -(1/m) * np.sum(y_vec * np.log(h) + (1-y_vec) * np.log(1-h))
            # show the cost variable with floating point precision of 4
            print(f"Iteration {i}: cost = {cost:.4f}")
    
    return weights

def predict(X, weights, threshold=0.5):
    X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])

    predictions = (X_with_bias @ weights)
    probabilities = sigmoid(predictions)
    verdicts = (probabilities >= threshold).astype(int)

    return verdicts

def get_accuracy(y, predictions, plot_c_mat=True, labels=[0, 1]):
    y_test_vec = np.array(y).reshape(-1, 1)
    if plot_c_mat:
        cm = confusion_matrix(y_test_vec, predictions)
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        disp.plot(cmap='Blues')

        plt.show()

    # use .flatten() to make sure both 'y' and 'predictions' arrays are of the same dimention
    print(f"Model score: {np.mean(predictions.flatten() == y_test_vec.flatten())}")

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
scaler.fit(X_train)

X_train_standardized, X_test_standardized = scaler.transform(X_train), scaler.transform(X_test)

feature_count = features.shape[1]
weights = train_logistic_regression(X_train_standardized, y_train, feature_count)
predictions = predict(X_test_standardized, weights)
print(get_accuracy(y_test, predictions))