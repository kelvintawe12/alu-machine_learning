#!/usr/bin/env python3
"""
Module for calculating the minor matrix of a matrix
"""
import importlib.util
import os


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
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is empty
    if len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    # Check if all rows have the same length
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Import determinant function from the same package
    try:
        # Try to import as a module from the current directory
        current_dir = os.path.dirname(__file__)
        spec = importlib.util.spec_from_file_location("determinant", os.path.join(current_dir, "0-determinant.py"))
        determinant_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(determinant_module)
        determinant_func = determinant_module.determinant
    except Exception:
        # Fallback: try to import directly
        try:
            from .determinant import determinant
            determinant_func = determinant
        except ImportError:
            raise ImportError("Could not import determinant function")

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
            row.append(determinant_func(submatrix))
        minor_matrix.append(row)

    return minor_matrix
