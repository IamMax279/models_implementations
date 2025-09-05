from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from math import sqrt, pow
import numpy as np

bunch = fetch_california_housing()

data, feature_names = bunch.data, bunch.feature_names
target, target_name = bunch.target, bunch.target_names

scaler = StandardScaler()
# Calculate and 'remember' the standard deviation and average of each feature
scaler.fit(data)

# Use standarization because knn is sensitive
standardized_data = scaler.transform(data)

new_features = [7.9123, 38.0, 7.1025, 1.045, 310.0, 2.65, 37.85, -122.20]
new_features_standardized = scaler.transform(np.array(new_features).reshape(1, -1))[0]

def knn_regression(k, features):
    if len(features) != len(data[0]):
        print(f"length of the new features array ({len(features)} doesn't equal the length \
        of the dataset rows ({len(data[0])}))")
        return

    distances = []

    for i, v in enumerate(standardized_data):
        cur_distance = sqrt(sum(pow((v[j] - features[j]), 2) for j in range(len(features))))
        distances.append([cur_distance, i])
    
    distances.sort(key=lambda x: x[0])
    k_distances = distances[:k]

    w_mean, w_sum = 0, 0
    for d, i in k_distances:
        # cecha * waga
        w_mean += target[i] * 1 / (d + 1e-8)
        w_sum += 1 / (d + 1e-8)
    
    return w_mean / w_sum

print(knn_regression(20, new_features_standardized))