#!/usr/bin/env python3
"""
Module for calculating polynomial integrals
"""


def poly_integral(poly, C=0):
    """
    Calculate the integral of a polynomial

    Args:
        poly (list): List of coefficients representing a polynomial
        C (int): Integration constant

    Returns:
        list: New list of coefficients representing the integral, or None if invalid
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if not isinstance(C, int):
        return None

    # Check if all elements are numbers
    for coeff in poly:
        if not isinstance(coeff, (int, float)):
            return None

    integral = [C]

    for i in range(len(poly)):
        power = i + 1
        if isinstance(poly[i], int) and isinstance(power, int):
            # Keep as integer if possible
            if poly[i] % power == 0:
                integral.append(poly[i] // power)
            else:
                integral.append(poly[i] / power)
        else:
            integral.append(poly[i] / power)

    # Remove trailing zeros
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
