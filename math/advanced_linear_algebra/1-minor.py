#!/usr/bin/env python3
"""
Module for calculating the minor matrix of a matrix
"""


def determinant(matrix):
    """
    Calculate the determinant of a matrix (helper function).
    """
    n = len(matrix)

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


def minor(matrix):
    """
    Calculate the minor matrix of a matrix.

    Args:
        matrix (list): A list of lists representing the matrix

    Returns:
        list: The minor matrix of the input matrix

    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square or is empty
    """
    # Check if matrix is a list of lists
    if (not isinstance(matrix, list) or
            not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is empty
    if len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    # Check if all rows have the same length (square matrix)
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Special case: 1x1 matrix
    if n == 1:
        return [[1]]

    # Calculate minor matrix
    minor_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            # Create submatrix by removing i-th row and j-th column
            submatrix = []
            for x in range(n):
                if x != i:
                    sub_row = []
                    for y in range(n):
                        if y != j:
                            sub_row.append(matrix[x][y])
                    submatrix.append(sub_row)

            # Calculate minor as determinant of submatrix
            row.append(determinant(submatrix))
        minor_matrix.append(row)

    return minor_matrix
