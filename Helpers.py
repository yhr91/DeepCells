"""
Use case: Deep learning for high content imaging
Description: Some helpers
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def one_hot(vec, cols = None):
    """ Take a vector of labels and converts it into a one hot 
    encoding
    """

    if cols is None:
        cols = len(np.unique(np.array(vec)))
    one_hot_vec = np.zeros([len(vec), cols])
    one_hot_vec[np.arange(len(vec)),vec] = 1
    return one_hot_vec

def mult_task_label(vec, d=4):
    """ Take a vector of int labels and converts it into a vector
    of binary lists
    """
    ret = []
    for v in vec:
        ret.append([i for i in ('{:0' + str(d) + 'b}').format(v)])
    return np.array(ret).astype('float')
