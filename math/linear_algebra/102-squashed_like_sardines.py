#!/usr/bin/env python3
"""
This module provides functions for concatenating matrices along specified axes.
"""


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis.

    Args:
        mat1: nested list for the first matrix.
        mat2: nested list for the second matrix.
        axis: axis along which to concatenate.

    Returns:
        new nested list with concatenated matrices, or None if cannot.
    """
    def get_shape(matrix):
        shape = []
        current = matrix
        while isinstance(current, list):
            shape.append(len(current))
            if current:
                current = current[0]
            else:
                break
        return shape

    shape1 = get_shape(mat1)
    shape2 = get_shape(mat2)
    if len(shape1) != len(shape2):
        return None
    for i in range(len(shape1)):
        if i != axis and shape1[i] != shape2[i]:
            return None

    def concat_recursive(m1, m2, axis, current_axis=0):
        if current_axis == axis:
            return m1 + m2
        else:
            if not isinstance(m1, list) or not isinstance(m2, list):
                return None
            if len(m1) != len(m2):
                return None
            result = [concat_recursive(a, b, axis, current_axis + 1)
                      for a, b in zip(m1, m2)]
            return result

    return concat_recursive(mat1, mat2, axis)
