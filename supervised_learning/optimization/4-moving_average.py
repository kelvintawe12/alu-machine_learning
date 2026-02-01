#!/usr/bin/env python3
"""
This module provides a function to calculate the weighted moving average of a data set with bias correction.
"""

def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set with bias correction.

    Parameters
    ----------
    data : list
        The data to calculate the moving average of.
    beta : float
        The weight used for the moving average.

    Returns
    -------
    list
        The moving averages of data.
    """
    m_avg = []
    v = 0
    for t, x in enumerate(data, 1):
        v = beta * v + (1 - beta) * x
        v_corr = v / (1 - beta ** t)
        m_avg.append(v_corr)
    return m_avg
