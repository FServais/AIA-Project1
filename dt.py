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
    number_sample = 2000
    number_of_samples__train_set = 150

    X, y = make_data(n_samples=number_sample)

    X_train, y_train = X[:number_of_samples__train_set], y[:number_of_samples__train_set]
    X_test, y_test = X[number_of_samples__train_set:], y[number_of_samples__train_set:]


    # 1.



    pass
