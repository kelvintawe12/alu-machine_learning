#!/usr/bin/env python3

def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix.

    Args:
        matrix: A 2D list representing the matrix.

    Returns:
        A new 2D list representing the transposed matrix.
    """
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
