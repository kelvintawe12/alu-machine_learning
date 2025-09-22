#!/usr/bin/env python3
"""
Module for calculating polynomial derivatives
"""


def poly_derivative(poly):
    """
    Calculate the derivative of a polynomial

    Args:
        poly (list): List of coefficients representing a polynomial

    Returns:
        list: New list of coefficients representing the derivative, or None if invalid
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    # Check if all elements are numbers
    for coeff in poly:
        if not isinstance(coeff, (int, float)):
            return None

    # Calculate derivative
    if len(poly) == 1:
        return [0]

    derivative = []
    for i in range(len(poly) - 1):
        derivative.append((len(poly) - 1 - i) * poly[i])

    # Remove trailing zeros
    while len(derivative) > 1 and derivative[-1] == 0:
        derivative.pop()

    return derivative
