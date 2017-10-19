# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    N=y.shape[0]
    txt=np.transpose(tx)    
    w=(np.linalg.inv(txt@tx))@txt@y
    return w