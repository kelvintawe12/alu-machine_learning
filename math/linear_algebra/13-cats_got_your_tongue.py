#!/usr/bin/env python3

import numpy as np

def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis.

    Args:
        mat1: A numpy.ndarray.
        mat2: A numpy.ndarray.
        axis: The axis along which to concatenate.

    Returns:
        A new numpy.ndarray with concatenated matrices.
    """
    return np.concatenate((mat1, mat2), axis=axis)
