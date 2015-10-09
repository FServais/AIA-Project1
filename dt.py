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

from data import make_data
from plot import plot_boundary


if __name__ == "__main__":
    # (Question 1) dt.py: Decision tree
    SAMPLE_NUMBER = 2000
    TRAIN_SET_SAMPLE_NUMBER = 150

    X, y = make_data(n_samples=SAMPLE_NUMBER)

    X_train, y_train = X[:TRAIN_SET_SAMPLE_NUMBER], y[:TRAIN_SET_SAMPLE_NUMBER]
    X_test, y_test = X[TRAIN_SET_SAMPLE_NUMBER:], y[TRAIN_SET_SAMPLE_NUMBER:]

    # 1.
    decisionTreeClassifier = DecisionTreeClassifier()
    decisionTreeClassifier.fit(X_train, y_train)
    y_dtc = decisionTreeClassifier.predict(X_test)

    # Plot
    plot_boundary("Reality", decisionTreeClassifier, X_test, y_test, title="Real data")
    plot_boundary("Result", decisionTreeClassifier, X_test, y_dtc, title="Result data")


    # 2.


    pass
