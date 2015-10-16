#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Only py3 / so that 2 / 3 = 0.66..
from __future__ import division
# Only py3 string encoding
from __future__ import unicode_literals
# Only py3 print
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from sklearn import cross_validation

from plot import plot_boundary
from utils import get_dataset

def compare(sampl_predict, sampl_real):
    """Compare two sample of the same size and return the number of difference.

    Parameters
    ----------
    sampl_predict : vector-like, shape (SAMPLE_NUMBER - TRAIN_SET_SAMPLE_NUM)
        prediction samples.
    sampl_real : vector-like, shape (SAMPLE_NUMBER - TRAIN_SET_SAMPLE_NUM)
        Real samples.

    Returns
    -------
    difference : int
        Number of difference between the two vectors
    """
    difference = 0
    for i in range(len(sampl_predict)):
        if sampl_predict[i] != sampl_real[i]:
            difference += 1

    return difference


if __name__ == "__main__":

    # (Question 1) dt.py: Decision tree
    SAMPLE_NUMBER = 2000
    TRAIN_SET_SAMPLE_NUM = 150

    X, y, random_state = get_dataset(SAMPLE_NUMBER)

    X_train, y_train = X[:TRAIN_SET_SAMPLE_NUM], y[:TRAIN_SET_SAMPLE_NUM]
    X_test, y_test = X[TRAIN_SET_SAMPLE_NUM:], y[TRAIN_SET_SAMPLE_NUM:]

    # 1.
    decisionTreeClassifier = DecisionTreeClassifier(random_state=random_state)
    decisionTreeClassifier.fit(X_train, y_train)
    y_dtc = decisionTreeClassifier.predict(X_test)

    # Plot
    plot_boundary("1-1-Ground-Truth", decisionTreeClassifier, X_test, y_test, title="Ground Truth data")
    plot_boundary("1-1-Prediction", decisionTreeClassifier, X_test, y_dtc, title="Prediction data")


    # 2.

    max_depths = [1, 2, 4, 8, 10, 50, 100, 150]
    error = []
    for max_depth in max_depths:
        decisionTreeClassifier = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
        decisionTreeClassifier.fit(X_train, y_train)
        y_dtc = decisionTreeClassifier.predict(X_test)
        
        error.append(compare(y_dtc, y_test))

        # Plot
        plot_boundary("Reality (%s)" % str(max_depth), decisionTreeClassifier, X_test, y_test, title="Real data (%s)" % str(max_depth))

    # 3.
    plt.figure()
    plt.title("Decision error induces by the model")
    plt.plot(max_depths, error)
    plt.xlabel("Value of max_depths")
    plt.xscale('log')
    plt.ylabel("Number of errors")
    plt.savefig("error_max_depth.pdf")
