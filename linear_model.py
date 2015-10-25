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
    X_pol, y_pol = get_dataset(SAMPLE_NUMBER)
    
    #1.
    
    X_train, y_train = X[:TRAIN_SET_SAMPLE_NUM], y[:TRAIN_SET_SAMPLE_NUM]
    X_test, y_test = X[TRAIN_SET_SAMPLE_NUM:], y[TRAIN_SET_SAMPLE_NUM:]
    
    ridgeClassifier = RidgeClassifier(alpha = 0)
    ridgeClassifier.fit(X_train, y_train)
    y_rc = ridgeClassifier.predict(X_test)
    
    # Plot
    plot_boundary("3-1-Ground-Truth", ridgeClassifier, X_test, y_test, title="Ground Truth data")
    plot_boundary("3-1-Prediction", ridgeClassifier, X_test, y_rc, title="Prediction data")

    #2.
    
    #Change in polar coordinates
    for i in range(len(X)):
        X_pol[i,0] = np.sqrt(X[i,0]**2+X[i,1]**2)        
        X_pol[i,1] = np.arctan2(X[i,1],X[i,0])
        
        if X_pol[i,1] <= X_pol[i,0] - 1./2*np.pi:
            
            if X_pol[i,1] <= X_pol[i,0] - 5./2*np.pi:
                X_pol[i,1] += 4*np.pi
                continue
            X_pol[i,1] += 2*np.pi
        
    X_train_pol, y_train_pol = X_pol[:TRAIN_SET_SAMPLE_NUM], y_pol[:TRAIN_SET_SAMPLE_NUM]
    X_test_pol, y_test_pol = X_pol[TRAIN_SET_SAMPLE_NUM:], y_pol[TRAIN_SET_SAMPLE_NUM:]
    
    ridgeClassifier = RidgeClassifier(alpha = 0)
    ridgeClassifier.fit(X_train_pol, y_train_pol)
    y_rc_pol = ridgeClassifier.predict(X_test_pol)
    
      
    
    plot_boundary("3-2-Ground-Truth", ridgeClassifier, X_test_pol, y_test_pol   , title="Ground Truth data")
    plot_boundary("3-2-Prediction", ridgeClassifier, X_test_pol, y_rc_pol, title="Prediction data")

    #3. 

    #Accuracy without transformation 
    
    error_without_transf = compare(y_rc, y_test)
    
    #Accuracy with transformation
    error_with_transf = compare(y_rc_pol, y_test_pol)

    #Results
    
    print("[Q3-3] Error percentage without transformation : {}%".format(error_without_transf*100/len(y_test)))
    print("[Q3-3] Error percentage with transformation :    {}%".format(error_with_transf*100/len(y_test)))
    
