#!/usr/bin/env python3
"""
Module for calculating the inverse of a matrix
"""


def inverse(matrix):
    """
    Calculate the inverse of a matrix.

    Args:
        matrix (list): A list of lists representing the matrix

    Returns:
        list: The inverse of the matrix, or None if matrix is singular

    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square or is empty
    """
    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is empty
    if len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    # Check if all rows have the same length
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Import required functions from the same package
    import determinant, adjugate

    # Calculate determinant
    det = determinant.determinant(matrix)

    # If determinant is zero, matrix is singular
    if det == 0:
        return None

    # Get adjugate matrix
    adj = adjugate.adjugate(matrix)

    # Calculate inverse: (1/det) * adjugate
    inverse_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(adj[i][j] / det)
        inverse_matrix.append(row)

    return inverse_matrix
