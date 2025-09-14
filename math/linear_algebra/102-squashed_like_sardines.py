#!/usr/bin/env python3

def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis.

    Args:
        mat1: A nested list representing the first matrix.
        mat2: A nested list representing the second matrix.
        axis: The axis along which to concatenate.

    Returns:
        A new nested list with concatenated matrices, or None if cannot concatenate.
    """
    def get_shape(m):
        if not isinstance(m, list):
            return []
        return [len(m)] + get_shape(m[0])

    if axis == 0:
        if isinstance(mat1, list) and isinstance(mat2, list):
            s1 = get_shape(mat1)
            s2 = get_shape(mat2)
            if s1[1:] != s2[1:]:
                return None
            return mat1 + mat2
        else:
            return None
    else:
        if isinstance(mat1, list) and isinstance(mat2, list):
            if len(mat1) != len(mat2):
                return None
            return [cat_matrices(a, b, axis - 1) for a, b in zip(mat1, mat2)]
        else:
            return None
