#!/usr/bin/env python3
"""
This module provides a function to calculate normalization (standardization) constants for a matrix.
"""
import numpy as np

def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix.

    Parameters
    ----------
    X : numpy.ndarray of shape (m, nx)
        The data to normalize.

    Returns
    -------
    mean : numpy.ndarray of shape (nx,)
        The mean of each feature.
    std : numpy.ndarray of shape (nx,)
        The standard deviation of each feature.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
