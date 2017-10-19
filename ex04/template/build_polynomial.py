# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""

    # this function should return the matrix formed
    # by applying the polynomial basis to the input data

    n=len(x)
    matr=np.ones(n,)[np.newaxis]

    for i in range(1,degree+1):
        xp=np.power(x,i)[np.newaxis]
        matr=np.append(matr,xp,axis=0)
    return np.transpose(matr)

	