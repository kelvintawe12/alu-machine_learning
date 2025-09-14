#!/usr/bin/env python3

def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise.

    Args:
        arr1: A list of ints/floats.
        arr2: A list of ints/floats.

    Returns:
        A new list with element-wise sum, or None if shapes differ.
    """
    if len(arr1) != len(arr2):
        return None
    return [a + b for a, b in zip(arr1, arr2)]
