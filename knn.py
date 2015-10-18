#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Only py3 / so that 2 / 3 = 0.66..
from __future__ import division
# Only py3 string encoding
from __future__ import unicode_literals
# Only py3 print
from __future__ import print_function

import numpy as np
import random

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from data import make_data
from plot import plot_boundary

from utils import get_dataset
from utils import get_random_state

import operator

def euclidean_distance(p1, p2):
    """Compute the Euclidean distance between 2 points

    Parameters
    ----------
    p1, p2: Numpy arrays, having the same dimensions.
            Points between which the distance is computed

    """
    return np.linalg.norm(p1 - p2)

class KNeighborsClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=1):
        """K-nearest classifier classifier

        Parameters
        ----------
        n_neighbors: int, optional (default=1)
            Number of neighbors to consider.

        """
        self.n_neighbors = n_neighbors
        self.train_samples = {}

    def fit(self, X, y):
        """Fit a k-nn model using the training set (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Parameter validation
        if self.n_neighbors < 0:
            raise ValueError("n_neighbors muster greater than 0, got %s"
                             % self.neighbors)

        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        # Save each sample (all variables) and the corresponding class
        # (var1, var2, ...)_i -> y_i
        for i in range(0,len(y)):
            self.train_samples[tuple(X[i])] = y[i]

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        y = []
        for test_sample in X:
            # Compute distances to each sample of the training set
            distances = {}
            for train_tuple, y_train in self.train_samples.items():
                distances[train_tuple] = euclidean_distance(np.array(train_tuple), test_sample)

            # Search k samples with the smallest distances
            sorted_distances = sorted(distances.items(), key=operator.itemgetter(1))
            k_nearest = sorted_distances[:self.n_neighbors]

            # Compute proportions for each classes
            classes = {}
            for neighb, _ in k_nearest:
                if self.train_samples[neighb] in classes:
                    classes[self.train_samples[neighb]] += 1
                else:
                    classes[self.train_samples[neighb]] = 1

            # Predict
            y.append(max(classes, key=classes.get))

        return y
        pass

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like, shape (n_query, n_features)
            Test samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        # TODO your code here.
        pass

if __name__ == "__main__":
    # (Question 2): K-nearest-neighbors

    TRAIN_SET_SAMPLE_NUM = 150
    X, y = get_dataset(2000)

    X_train, y_train = X[:TRAIN_SET_SAMPLE_NUM], y[:TRAIN_SET_SAMPLE_NUM]
    X_test, y_test = X[TRAIN_SET_SAMPLE_NUM:], y[TRAIN_SET_SAMPLE_NUM:]

    knc = KNeighborsClassifier(n_neighbors=10)
    knc.fit(X_train, y_train)
    y_predict = knc.predict(X_test)

    n_errors = sum([1 if y_test[i] != y_predict[i] else 0 for i in range(0, len(y_test))])

    print("Error percentage : {}%".format(n_errors/len(X_test)))

        
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
