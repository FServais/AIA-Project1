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
from sklearn.linear_model import RidgeClassifier

from data import make_data
from plot import plot_boundary

from utils import get_dataset
from utils import get_random_state
from utils import compare



if __name__ == "__main__":
    # (Question 3) ols.py: Ordinary least square
    
    SAMPLE_NUMBER = 2000
    TRAIN_SET_SAMPLE_NUM = 150
    X, y = get_dataset(SAMPLE_NUMBER)

    X_train, y_train = X[:TRAIN_SET_SAMPLE_NUM], y[:TRAIN_SET_SAMPLE_NUM]
    X_test, y_test = X[TRAIN_SET_SAMPLE_NUM:], y[TRAIN_SET_SAMPLE_NUM:]
    
    #1.
    
    ridgeClassifier = RidgeClassifier(alpha = 0)
    ridgeClassifier.fit(X_train, y_train)
    y_rc = ridgeClassifier.predict(X_test)
    
    #plot
    plot_boundary("3-1-Ground-Truth", ridgeClassifier, X_test, y_test, title="Ground Truth data")
    plot_boundary("3-1-Prediction", ridgeClassifier, X_test, y_rc, title="Prediction data")
    
    #TODO: 
    #2.
    

    #3. 

    #Accuracy without transformation 
    
    error_without_transf = compare(y_rc, y_test)
    
    
    #Accuracy with transformation
    
    
    #Results
    
    print("[Q3-3] Error percentage without transformation : {}%".format(error_without_transf*100/len(y_test)))
    
