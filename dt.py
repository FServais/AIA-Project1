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
from utils import get_random_state

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

    X, y = get_dataset(SAMPLE_NUMBER)
    random_state = get_random_state()

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

    max_depths = [1] + [i for i in range(20,TRAIN_SET_SAMPLE_NUM,30)] + [TRAIN_SET_SAMPLE_NUM]
    for max_depth in max_depths:
        decisionTreeClassifier = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
        decisionTreeClassifier.fit(X_train, y_train)
        y_dtc = decisionTreeClassifier.predict(X_test)

        # Plot
        plot_boundary("Reality (%s)" % str(max_depth), decisionTreeClassifier, X_test, y_test, title="Real data (%s)" % str(max_depth))

    # 3.

    max_depths = [1] + [i for i in range(20,TRAIN_SET_SAMPLE_NUM,5)] + [TRAIN_SET_SAMPLE_NUM]
    accuracy = []
    for max_depth in max_depths:
        decisionTreeClassifier = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
        decisionTreeClassifier.fit(X_train, y_train)
        y_dtc = decisionTreeClassifier.predict(X_test)

        accuracy.append((SAMPLE_NUMBER-TRAIN_SET_SAMPLE_NUM - compare(y_dtc, y_test))/(SAMPLE_NUMBER-TRAIN_SET_SAMPLE_NUM))

    plt.figure()
    plt.title("Accuracy on the testing set induced by the model")
    plt.plot(max_depths, accuracy)
    plt.xlabel("Value of max_depth")
    plt.ylabel("Accuracy (%)")
    plt.savefig("accuracy_max_depth.pdf")
