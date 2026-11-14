#!/usr/bin/env python3
"""Local regression: for each test point, a tricube-weighted linear model
fitted on its nearest training points (a `frac` share of the data).

@author: sibirbil
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression


class SklearnEstimator:
    """Interface with fit and predict."""
    def fit(self, X: np.array, y: np.array):
        raise NotImplementedError("Needs to implement fit(X, y)")

    def predict(self, X0: np.array):
        raise NotImplementedError("Needs to implement predict(X, y)")

class LocalRegression(RegressorMixin, BaseEstimator, SklearnEstimator):
    """
        Parameters
        ----------
            frac : Fraction of total samples used for local fit
            
        Source
        ------
        Elements of Statistical Learning, T. Hastie, R. Tibshirani, J. Friedman, 
        2nd Edition, Springer (Section 6.3)
    """
    def __init__(self, frac=0.2):
        
        self.X = None
        self.y = None
        self.frac = frac
        self.is_fitted_ = False

    def fit(self, X: np.array, y: np.array):
        
        self.X = X
        self.y = y
        
        self.is_fitted_ = True

        return self

    def predict(self, X0: np.array):
        
        if self.X is None:
            raise ValueError("You need to fit the LocalRegression model first")

        N = len(self.X)
        n_neighbors = int(np.ceil(self.frac * N))
        N0 = len(X0)    
        y0 = np.zeros(N0)
    
        if (X0.shape == (N0,)):
            X0 = X0.reshape(-1,1)   
        
        h = [np.sort(np.linalg.norm(self.X - X0[i], axis=1))[n_neighbors] for i in range(N0)]
        
        for i in range(N0):
            if (h[i] < 1.0e-6):
                # Just to avoid division by zero
                t = np.linalg.norm(self.X - X0[i], axis=1)/(h[i]+1.0e-6)
            else:
                t = np.linalg.norm(self.X - X0[i], axis=1)/h[i]
            mask = t <= 1.0
            Kn = np.zeros(N)
            Kn[mask] = (1.0 - t[mask] ** 3.0) ** 3.0
            # Weighted least squares through sample_weight, so the intercept is
            # weighted too and points outside the kernel get zero weight.
            # predict returns a one-element array; [0] takes the scalar.
            y0[i] = (
                LinearRegression()
                .fit(self.X, self.y, sample_weight=Kn)
                .predict(X0[i].reshape(1, -1))[0]
            )

        return y0