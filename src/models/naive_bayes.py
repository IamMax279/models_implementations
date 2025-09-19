import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def train_naive_bayes(X, y):
    cv = CountVectorizer()
    X_mat = cv.fit_transform(X).toarray()

    class_counts = np.bincount(y)
    priors = class_counts / len(y)

    # spam - 1, ham - 0 (2 classes)
    word_counts_per_class = np.zeros((2, X_mat.shape[1]), dtype=int)
    for i in range(2):
        # y == i creates a boolean mask (e.g. [True, False, True, ...])
        # so ultimately we only select X_mat rows where y[j] == True
        word_counts_per_class[i] = X_mat[y == i].sum(axis=0)
    
    total_words_per_class = word_counts_per_class.sum(axis=1)
    # + 1 is Laplace smoothing (to avoid zero probability)
    # [:, None] adds a dimension to array - e.g. [1, 2, 3] becomes
    # [[1],
    #  [2],
    #  [3]]
    likelihoods = (word_counts_per_class + 1) / (total_words_per_class[:, None] + X_mat.shape[1])
    return cv, priors, likelihoods

def predict(X, cv, priors, likelihoods):
    X_mat = cv.transform(X).toarray()
    
    # when calculating likelihoods, we might be multiplying small numbers like
    # 0.001 * 0.002 * ... - hence we use np.log() because the logarithm
    # will convert multiplication to addition and thus, we'll avoid underflow
    log_likelihoods = np.log(likelihoods)
    log_priors = np.log(priors)
    posteriors = X_mat @ log_likelihoods.T + log_priors

    predictions = np.argmax(posteriors, axis=1)
    return predictions

def get_accuracy(y, predictions, plot_c_mat=True, labels=[0, 1]):
    y_vec = np.array(y)
    if plot_c_mat:
        cm = confusion_matrix(y_vec, predictions.flatten())
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)

        disp.plot(cmap='Blues')
        plt.show()
    
    return np.mean(y_vec == predictions.flatten())
    # alternatively:
    # return (y_vec == predictions.flatten()).sum(axis=0) / y_vec.shape[0]

df = pd.read_csv('src/data/spam.csv', sep=',')
df['Category'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Category'], test_size=0.25, random_state=42)

cv, priors, likelihoods = train_naive_bayes(X_train, y_train)
predictions = predict(X_test, cv, priors, likelihoods)
score = get_accuracy(y_test, predictions)