#!/usr/bin/env python3

def trim_me_down(matrix):
    """
    Extract middle columns.
    """
    return [[row[2], row[3]] for row in matrix]
