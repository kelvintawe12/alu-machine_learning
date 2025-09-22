#!/usr/bin/env python3
"""
Module for calculating the determinant of a matrix
"""


def determinant(matrix):
    """
    Calculate the determinant of a matrix.

    Args:
        matrix (list): A list of lists representing the matrix

    Returns:
        float: The determinant of the matrix

    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square
    """
    # Check if matrix is a list of lists
    if (not isinstance(matrix, list) or
            not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is empty
    if len(matrix) == 0:
        raise ValueError("matrix must be a square matrix")

    # Check if all rows have the same length
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        # Special case: 1x0 matrix [[]] should return 1
        if n == 1 and len(matrix[0]) == 0:
            return 1
        raise ValueError("matrix must be a square matrix")

    # Base case: 0x0 matrix
    if n == 0:
        return 1

    # Base case: 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # Base case: 2x2 matrix
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive case: n x n matrix (n > 2)
    det = 0
    for j in range(n):
        # Create submatrix by removing first row and j-th column
        submatrix = []
        for i in range(1, n):
            row = []
            for k in range(n):
                if k != j:
                    row.append(matrix[i][k])
            submatrix.append(row)

        # Calculate cofactor: (-1)^(0+j) * det(submatrix)
        cofactor = (-1) ** j * determinant(submatrix)
        det += matrix[0][j] * cofactor

    return det
