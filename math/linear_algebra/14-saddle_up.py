#!/usr/bin/env python3
"""
Module for matrix multiplication using numpy.
"""

import numpy as np


def np_matmul(mat1, mat2):
    """
    Performs matrix multiplication.

    Args:
        mat1: A numpy.ndarray.
        mat2: A numpy.ndarray.

    Returns:
        A new numpy.ndarray representing the product.
    """
    return mat1 @ mat2
