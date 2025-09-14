#!/usr/bin/env python3

def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specific axis.

    Args:
        mat1: A 2D list of ints/floats.
        mat2: A 2D list of ints/floats.
        axis: The axis along which to concatenate (0 or 1).

    Returns:
        A new 2D list with concatenated matrices, or None if cannot concatenate.
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    else:
        return None
