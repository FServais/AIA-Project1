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


class KNeighborsClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=1):
        """K-nearest classifier classifier

        Parameters
        ----------
        n_neighbors: int, optional (default=1)
            Number of neighbors to consider.

        """
        self.n_neighbors = n_neighbors

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

        # TODO your code here.

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
        # TODO your code here.
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
    
    SAMPLE_NUMBER = 200
    K = 15 #Random ?
    x_prime = [0,0] #Random ?
    
    dist = [0]*SAMPLE_NUMBER
    dist_temp = [0]*SAMPLE_NUMBER
    index_neighbor = [0]*K
    
    # FIRST PART
    X, y = make_data(n_samples=SAMPLE_NUMBER)
    
    #1.Compute all the distance with the test value
    for i in range(SAMPLE_NUMBER):   
        dist[i] = np.linalg.norm(X[i]-x_prime)
        dist_temp[i] = np.linalg.norm(X[i]-x_prime)
    
    #2.Find the index of the K-nearest-Neighbors
    for i in range(K):
        min_temp =  min(dist_temp)        
        index_neighbor[i] = dist_temp.index(min_temp)
        dist_temp[index_neighbor[i]] = float('inf')
        
    #3.Compute the proportion of sample of each class among 
    #the k-nearest-neighbor
    #Works only if there are two classes
    class0 = 0
    class1 = 0
    for i in range(K):
        if y[index_neighbor[i]]:
            class1 +=1
        else:
            class0 +=1
            
    #4.Prediction
    
    if class0 > class1:
        y_prime = 0
    elif class0 < class1:
        y_prime = 1
    else:
        #if the probability is the same 
        y_prime = random.randint(0,1)
        
        
        
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
