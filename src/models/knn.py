from sklearn.datasets import fetch_california_housing
from math import sqrt, pow

bunch = fetch_california_housing()

data, feature_names = bunch.data, bunch.feature_names
target, target_name = bunch.target, bunch.target_names

new_features = [7.9123, 38.0, 7.1025, 1.045, 310.0, 2.65, 37.85, -122.20]

def knn_regression(k, features):
    if len(features) != len(data[0]):
        print(f"length of the new features array ({len(features)} doesn't equal the length \
        of the dataset rows ({len(data[0])}))")
        return

    distances = []

    for i, v in enumerate(data):
        cur_distance = sqrt(sum(pow((v[j] - features[j]), 2) for j in range(len(features))))
        distances.append([cur_distance, i])
    
    distances.sort(key=lambda x: x[0])
    k_distances = distances[:k]

    mean = 0
    for d, i in k_distances:
        mean += target[i]
    
    return mean / k

print(knn_regression(20, new_features))