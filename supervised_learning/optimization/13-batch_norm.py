#!/usr/bin/env python3
"""
This module provides a function to normalize an unactivated output.
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using batch normalization.

    
    -------
    np.ndarray
        The normalized Z matrix.
    """
    mean = np.mean(Z, axis=0, keepdims=True)
    var = np.var(Z, axis=0, keepdims=True)
    Z_norm = (Z - mean) / np.sqrt(var + epsilon)
    return gamma * Z_norm + beta
