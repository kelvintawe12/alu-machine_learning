#!/usr/bin/env python3
"""
Module for determining the definiteness of a matrix
"""
import numpy as np


def definiteness(matrix):
    """
    Calculate the definiteness of a matrix.

    Args:
        matrix (numpy.ndarray): A numpy array representing the matrix

    Returns:
        str: The definiteness classification or None if not applicable

    Raises:
        TypeError: If matrix is not a numpy.ndarray
    """
    # Check if matrix is a numpy array
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check if matrix is empty
    if matrix.size == 0:
        return None

    # Check if matrix is 2D and square
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    # Check if matrix contains only finite numbers
    if not np.all(np.isfinite(matrix)):
        return None

    # Check if matrix is symmetric (required for definiteness)
    if not np.allclose(matrix, matrix.T, rtol=1e-05, atol=1e-08):
        return None

    # Calculate eigenvalues
    try:
        eigenvals = np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        return None

    # Use a small tolerance for zero comparison
    tol = 1e-8

    # Check for positive definite: all eigenvalues > 0
    if np.all(eigenvals > tol):
        return "Positive definite"

    # Check for positive semi-definite: all eigenvalues >= 0
    if np.all(eigenvals >= -tol):
        return "Positive semi-definite"

    # Check for negative definite: all eigenvalues < 0
    if np.all(eigenvals < -tol):
        return "Negative definite"

    # Check for negative semi-definite: all eigenvalues <= 0
    if np.all(eigenvals <= tol):
        return "Negative semi-definite"

    # Check for indefinite: has both positive and negative eigenvalues
    if np.any(eigenvals > tol) and np.any(eigenvals < -tol):
        return "Indefinite"

    # If none of the above, return None
    return None
