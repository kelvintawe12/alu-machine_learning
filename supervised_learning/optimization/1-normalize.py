#!/usr/bin/env python3
"""
This module provides a function to normalize (standardize) a matrix.
"""
import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix.

    Parameters
    ----------
    X : numpy.ndarray of shape (d, nx)
        The data to normalize.
    m : numpy.ndarray of shape (nx,)
        The mean of all features of X.
    s : numpy.ndarray of shape (nx,)
        The standard deviation of all features of X.

    Returns
    -------
    X_norm : numpy.ndarray
        The normalized X matrix.
    """
    return (X - m) / s
