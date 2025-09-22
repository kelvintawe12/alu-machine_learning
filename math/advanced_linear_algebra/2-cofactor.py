#!/usr/bin/env python3
"""
Module for calculating the cofactor matrix of a matrix
"""


def cofactor(matrix):
    """
    Calculate the cofactor matrix of a matrix.

    Args:
        matrix (list): A list of lists representing the matrix

    Returns:
        list: The cofactor matrix of the input matrix

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

    # Import minor function from the same package
    import minor

    # Calculate cofactor matrix
    cofactor_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            # Get minor value
            minor_val = minor.minor(matrix)[i][j]
            # Apply cofactor sign: (-1)^(i+j)
            cofactor_val = (-1) ** (i + j) * minor_val
            row.append(cofactor_val)
        cofactor_matrix.append(row)

    return cofactor_matrix
