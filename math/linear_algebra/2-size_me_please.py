#!/usr/bin/env python3

def matrix_shape(matrix):
    """
    Calculates the shape of a matrix.

    Args:
        matrix: A nested list representing the matrix.

    Returns:
        A list of integers representing the shape of the matrix.
    """
    if not isinstance(matrix, list):
        return []
    if not matrix:
        return [0]
    shape = [len(matrix)]
    shape.extend(matrix_shape(matrix[0]))
    return shape
