#!/usr/bin/env python3
"""
This module provides a function to shuffle data points in two matrices the same way.
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Parameters
    ----------
    X : numpy.ndarray of shape (m, nx)
        The first matrix to shuffle.
    Y : numpy.ndarray of shape (m, ny)
        The second matrix to shuffle.

    Returns
    -------
    X_shuffled : numpy.ndarray
        The shuffled X matrix.
    Y_shuffled : numpy.ndarray
        The shuffled Y matrix.
    """
    perm = np.random.permutation(X.shape[0])
    return X[perm], Y[perm]
