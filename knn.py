#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Only py3 / so that 2 / 3 = 0.66..
from __future__ import division
# Only py3 string encoding
from __future__ import unicode_literals
# Only py3 print
from __future__ import print_function
from matplotlib.colors import ListedColormap

import numpy as np
import random
from sklearn import grid_search
import sklearn

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.neighbors import KNeighborsClassifier

from data import make_data
from plot import plot_boundary

from utils import get_dataset
from utils import get_random_state
from utils import compare

import operator

from matplotlib import pyplot as plt


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
    SAMPLE_NUMBER = 2000
    TRAIN_SET_SAMPLE_NUM = 150
    X, y = get_dataset(SAMPLE_NUMBER)

    X_train, y_train = X[:TRAIN_SET_SAMPLE_NUM], y[:TRAIN_SET_SAMPLE_NUM]
    X_test, y_test = X[TRAIN_SET_SAMPLE_NUM:], y[TRAIN_SET_SAMPLE_NUM:]

    # 1.
    knc = KNeighborsClassifier(n_neighbors=1)
    knc.fit(X_train, y_train)
    y_predict = knc.predict(X_test)

    n_errors = sum([1 if y_test[i] != y_predict[i] else 0 for i in range(0, len(y_test))])
    print("[Q2-1] Error percentage : {}%".format(n_errors/len(X_test)))

    # 2.
    oneNN = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
    oneNN.fit(X_train, y_train)
    y_predict = oneNN.predict(X_test)

    plot_boundary("2-2-Ground-Truth", oneNN, X_test, y_test, title="Ground Truth data")
    plot_boundary("2-2-Prediction", oneNN, X_test, y_predict, title="Prediction data")

    n_errors = sum([1 if y_test[i] != y_predict[i] else 0 for i in range(0, len(y_test))])
    print("[Q2-2] Error percentage : {}%".format(n_errors/len(X_test)))

    # 3.
    n_neighbors = [1, 2, 4, 7, 10, 30, 90, 150]
    for n in n_neighbors:
        nearest_neighb_class = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n)
        nearest_neighb_class.fit(X_train, y_train)
        y_predict = nearest_neighb_class.predict(X_test)

        plot_boundary("2-3-Prediction-%s" % str(n), nearest_neighb_class, X_test, y_predict, title="Prediction data")

    plot_boundary("2-3-Training-set", nearest_neighb_class, X_train, y_train, title="Training set boundaries")
    # 4.
    n_neighbors = [i for i in range(1,TRAIN_SET_SAMPLE_NUM)]
    error_training = {}
    error_testing = {}

    for n in n_neighbors:
        nearest_neighb_class = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n)
        nearest_neighb_class.fit(X_train, y_train)
        y_predict = nearest_neighb_class.predict(X_test)
        y_train_predict = nearest_neighb_class.predict(X_train)

        error_training[n] = compare(y_train, y_train_predict)/len(y_train)
        error_testing[n] = compare(y_test, y_predict)/len(y_test)

    plt.figure()
    plt.title("Error on the learning and testing sets induced by the model")
    tr, = plt.plot(n_neighbors, list(error_training.values()), label="Training set")
    ts, = plt.plot(n_neighbors, list(error_testing.values()), label="Testing set")
    plt.legend(handles=[tr, ts])
    plt.xlabel("Value of n_neighbors")
    plt.ylabel("Error (%)")
    plt.savefig("2-4-error_n_neighbors.pdf")

    # 5.
    N_FOLDS = 10
    nearest_neighb_class = sklearn.neighbors.KNeighborsClassifier()
    parameters = {'n_neighbors': [i for i in range(1, (N_FOLDS-1)*TRAIN_SET_SAMPLE_NUM//N_FOLDS)]}
    grid = grid_search.GridSearchCV(estimator=nearest_neighb_class, param_grid=parameters, cv=N_FOLDS)

    grid.fit(X_train, y_train)

    print("[Q5] Max score for N = {}".format(grid.best_estimator_.n_neighbors))