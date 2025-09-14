#!/usr/bin/env python3

def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.

    Args:
        mat1: A 2D list of ints/floats.
        mat2: A 2D list of ints/floats.

    Returns:
        A new 2D list with element-wise sum, or None if shapes differ.
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [[a + b for a, b in zip(row1, row2)] for row1, row2 in zip(mat1, mat2)]
