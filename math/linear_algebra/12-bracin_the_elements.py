#!/usr/bin/env python3
"""
Module for performing element-wise operations on numpy arrays.
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication, and division.

    Args:
        mat1: A numpy.ndarray.
        mat2: A numpy.ndarray or scalar.

    Returns:
        A tuple with the element-wise sum, difference, product, and quotient.
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
