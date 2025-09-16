from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import numpy as np

bunch = fetch_california_housing()

data, feature_names = bunch.data, bunch.feature_names
target, target_name = bunch.target, bunch.target_names

scaler = StandardScaler()
# Calculate and 'remember' the standard deviation and average of each feature
scaler.fit(data)

standardized_data = scaler.transform(data)

new_features = [7.9123, 38.0, 7.1025, 1.045, 310.0, 2.65, 37.85, -122.20]
new_features_standardized = scaler.transform(np.array(new_features).reshape(1, -1))[0]

def knn_regression(k, features):
    if len(features) != len(data[0]):
        print(f"length of the new features array ({len(features)} doesn't equal the length \
        of the dataset rows ({len(data[0])}))")
        return

    distances = np.linalg.norm(standardized_data - features, axis=1)
    
    # get k indexes that would sort the arrary in increasing order
    k_idx = np.argsort(distances)[:k]
    k_distances = distances[k_idx]
    k_targets = target[k_idx]
    
    weights = 1 / (k_distances + 1e-8)
    w_mean = k_targets @ weights
    w_sum = np.sum(weights)
    return w_mean / w_sum

print(knn_regression(20, new_features_standardized))