import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

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
    likelihood = (word_counts_per_class + 1) / (total_words_per_class[:, None] + X_mat.shape[1])
    

df = pd.read_csv('src/data/spam.csv', sep=',')
df['Category'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Category'], test_size=0.25, random_state=42)

train_naive_bayes(X_train, y_train)