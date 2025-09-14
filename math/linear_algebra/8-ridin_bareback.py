#!/usr/bin/env python3

def mat_mul(mat1, mat2):
    """
    Performs matrix multiplication.

    Args:
        mat1: A 2D list representing the first matrix.
        mat2: A 2D list representing the second matrix.

    Returns:
        A new 2D list representing the product, or None if cannot multiply.
    """
    if len(mat1[0]) != len(mat2):
        return None
    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            sum_val = 0
            for k in range(len(mat2)):
                sum_val += mat1[i][k] * mat2[k][j]
            row.append(sum_val)
        result.append(row)
    return result
