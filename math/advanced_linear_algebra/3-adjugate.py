#!/usr/bin/env python3
"""
Module for calculating the adjugate matrix of a matrix
"""


def adjugate(matrix):
    """
    Calculate the adjugate matrix of a matrix.

    Args:
        matrix (list): A list of lists representing the matrix

    Returns:
        list: The adjugate matrix of the input matrix

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

    # Import cofactor function from the same package
    import cofactor

    # Get cofactor matrix
    cofactor_matrix = cofactor.cofactor(matrix)

    # Transpose the cofactor matrix to get adjugate
    adjugate_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(cofactor_matrix[j][i])
        adjugate_matrix.append(row)

    return adjugate_matrix
