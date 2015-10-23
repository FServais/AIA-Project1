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
from sklearn import grid_search

from plot import plot_boundary
from utils import get_dataset
from utils import get_random_state
from utils import compare

if __name__ == "__main__":

    # (Question 1) dt.py: Decision tree
    SAMPLE_NUMBER = 2000
    TRAIN_SET_SAMPLE_NUM = 150

    X, y = get_dataset(SAMPLE_NUMBER)

    X_train, y_train = X[:TRAIN_SET_SAMPLE_NUM], y[:TRAIN_SET_SAMPLE_NUM]
    X_test, y_test = X[TRAIN_SET_SAMPLE_NUM:], y[TRAIN_SET_SAMPLE_NUM:]

    # 1.
    decisionTreeClassifier = DecisionTreeClassifier(random_state=get_random_state())
    decisionTreeClassifier.fit(X_train, y_train)
    y_dtc = decisionTreeClassifier.predict(X_test)

    # Plot
    plot_boundary("1-1-Ground-Truth", decisionTreeClassifier, X_test, y_test, title="Ground Truth data")
    plot_boundary("1-1-Prediction", decisionTreeClassifier, X_test, y_dtc, title="Prediction data")


    # 2.
    max_depths = [i for i in range(1, 14)]
    training_scores = []
    for max_depth in max_depths:
        decisionTreeClassifier = DecisionTreeClassifier(random_state=get_random_state(), max_depth=max_depth)
        decisionTreeClassifier.fit(X_train, y_train)
        y_dtc = decisionTreeClassifier.predict(X_test)

        # Plot
        plot_boundary("1-2-Max-Depth_%s" % str(max_depth), decisionTreeClassifier, X_test, y_test, title="Real data with max_depth = %s" % str(max_depth))

        training_scores.append(decisionTreeClassifier.score(X_train,y_train))

    print("[Q2] Score = 1 for the training set with max_depth = {}".format(training_scores.index(1.0)+1))

    # 3.
    max_depths = [i for i in range(1, 20)]
    error_training = {}
    error_testing = {}

    for max_depth in max_depths:
        decisionTreeClassifier = DecisionTreeClassifier(random_state=get_random_state(), max_depth=max_depth)
        decisionTreeClassifier.fit(X_train, y_train)
        # Learning sample
        y_train_predict = decisionTreeClassifier.predict(X_train)
        error_training[max_depth] = compare(y_train_predict, y_train)*100/len(y_train)

        # Testing sample
        y_dtc = decisionTreeClassifier.predict(X_test)
        error_testing[max_depth] = compare(y_dtc, y_test)*100/len(y_test)

    min_error_depth = min(error_training, key=error_training.get)
    print("[Q3 - Training set] Min error for depth = {}".format(min_error_depth))

    min_error_depth = min(error_testing, key=error_testing.get)
    print("[Q3 - Testing set] Min error for depth = {}".format(min_error_depth))

    plt.figure()
    plt.title("Error on the learning and testing sets induced by the model")
    tr, = plt.plot(max_depths, list(error_training.values()), label="Training set")
    ts, = plt.plot(max_depths, list(error_testing.values()), label="Testing set")
    plt.legend(handles=[tr, ts])
    plt.xlabel("Value of max_depth")
    plt.ylabel("Error (%)")
    plt.savefig("1-3-error_max_depth.pdf")

    # 4.
    N_FOLDS = 10
    decisionTreeClassifier = DecisionTreeClassifier(random_state=get_random_state())
    parameters = {'max_depth': [i for i in range(1, TRAIN_SET_SAMPLE_NUM+1)]}
    grid = grid_search.GridSearchCV(estimator=decisionTreeClassifier, param_grid=parameters, cv=N_FOLDS)

    grid.fit(X_train, y_train)

    print("[Q4] Max score for depth = {}".format(grid.best_estimator_.max_depth))
