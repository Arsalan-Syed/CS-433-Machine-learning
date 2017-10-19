# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    D = tx[0].size

    # w = (Xt + λ' * I)^(-1) * Xt * y
    lambdap = lambda_ * (2 * len(y))
    A = tx.T.dot(tx) + lambda_ * np.eye(D)
    w = np.linalg.inv(A).dot(tx.T.dot(y))

    loss = compute_mse_loss(y, tx, w)

    return w, loss

def compute_loss(y, tx, w):
    return compute_mse_loss(y, tx, w)

def compute_mse_loss(y, tx, w):
    """
    Calculate the loss for mse
    """
    e = y - np.dot(tx, w)
    N = e.size
    return (1./(2*N)) * np.sum(np.square(e))