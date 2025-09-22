#!/usr/bin/env python3
"""
Module for calculating the sum of squares from 1 to n
"""


def summation_i_squared(n):
    """
    Calculate the sum of squares from 1 to n using the formula n(n+1)(2n+1)/6

    Args:
        n (int): The stopping condition

    Returns:
        int: The sum of squares, or None if n is not valid
    """
    if not isinstance(n, int) or n < 1:
        return None

    return n * (n + 1) * (2 * n + 1) // 6
