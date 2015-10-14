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
    
    #Compare two sample of the same size.
    #Return the number of difference between these two.
    def compare(sampl_predict, sampl_real):
        difference = 0
        for i in range(len(sampl_predict)):
            if sampl_predict[i] != sampl_real[i]:
                difference += 1
    
        return difference
                
            
    
    
    # (Question 1) dt.py: Decision tree
    
    SAMPLE_NUMBER = 2000
    TRAIN_SET_SAMPLE_NUM = 150

    X, y = make_data(n_samples=SAMPLE_NUMBER)

    X_train, y_train = X[:TRAIN_SET_SAMPLE_NUM], y[:TRAIN_SET_SAMPLE_NUM]
    X_test, y_test = X[TRAIN_SET_SAMPLE_NUM:], y[TRAIN_SET_SAMPLE_NUM:]

    # 1.
    decisionTreeClassifier = DecisionTreeClassifier()
    decisionTreeClassifier.fit(X_train, y_train)
    y_dtc = decisionTreeClassifier.predict(X_test)

    # Plot
    plot_boundary("1-1-Ground-Truth", decisionTreeClassifier, X_test, y_test, title="Ground Truth data")
    plot_boundary("1-1-Prediction", decisionTreeClassifier, X_test, y_dtc, title="Prediction data")


    # 2.
    i = 0
    
    max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100, 1000, 10000, 100000]#,None]
    error = [0]*len(max_depths)  
    for max_depth in max_depths:
        decisionTreeClassifier = DecisionTreeClassifier(max_depth=max_depth)
        decisionTreeClassifier.fit(X_train, y_train)
        y_dtc = decisionTreeClassifier.predict(X_test)
        
        error[i] = compare(y_dtc,y_test)
        i += 1

        # Plot
        plot_boundary("Reality (%s)" % str(max_depth), decisionTreeClassifier, X_test, y_test, title="Real data (%s)" % str(max_depth))
        

        

    pass

    plt.figure();
    plt.title("Decision error induces by the model");
    plt.plot(max_depths,error)
    plt.xlabel("Value of max_depths");
    plt.xscale('log');
    plt.ylabel("Number of errors"); 
    plt.savefig("max_depth.pdf")
    
    