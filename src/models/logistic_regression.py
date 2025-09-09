import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def logistic_regression(X, y, feature_count):
    X_mat = np.array(X).reshape(-1, feature_count)

df = pd.read_csv('src/data/diabetes.csv', sep=',')

values_to_map = {
    'Male': 1,
    'Female': 0
}

# TODO: also map the 'smoking_history' column to numerical values
df['gender'] = df['gender'].map(values_to_map)

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

feature_count = len(np.array(features)[0])
logistic_regression(X_train, y_train, feature_count)