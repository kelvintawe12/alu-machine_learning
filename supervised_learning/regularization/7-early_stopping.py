#!/usr/bin/env python3
"""
Determines if you should stop gradient descent early (early stopping).
"""

def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early.

    Parameters
    ----------
    cost : float
        Current validation cost of the neural network.
    opt_cost : float
        Lowest recorded validation cost of the neural network.
    threshold : float
        Threshold used for early stopping.
    patience : int
        Patience count used for early stopping.
    count : int
        Count of how long the threshold has not been met.

    Returns
    -------
    tuple
        (should_stop, updated_count)
    """
    if cost < opt_cost - threshold:
        return False, 0
    count += 1
    if count >= patience:
        return True, count
    return False, count
