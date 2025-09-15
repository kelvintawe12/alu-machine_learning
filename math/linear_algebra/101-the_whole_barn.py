#!/usr/bin/env python3

"""
This module provides functions for matrix operations in linear algebra.
"""


def add_matrices(mat1, mat2):
    """
    Adds two matrices element-wise.

    Args:
        mat1: A nested list representing the first matrix.
        mat2: A nested list representing the second matrix.

    Returns:
        A new nested list with element-wise sum, or None if shapes differ.
    """
    if isinstance(mat1, list) and isinstance(mat2, list):
        if len(mat1) != len(mat2):
            return None
        return [add_matrices(a, b) for a, b in zip(mat1, mat2)]
    elif not isinstance(mat1, list) and not isinstance(mat2, list):
        return mat1 + mat2
    else:
        return None
