#!/usr/bin/env python3

def np_slice(matrix, axes={}):
    """
    Slices a matrix along specific axes.

    Args:
        matrix: A numpy.ndarray.
        axes: A dictionary where keys are axes and values are tuples for slicing.

    Returns:
        A new numpy.ndarray sliced according to axes.
    """
    slices = [slice(None)] * matrix.ndim
    for axis, sl in axes.items():
        slices[axis] = slice(*sl)
    return matrix[tuple(slices)]
