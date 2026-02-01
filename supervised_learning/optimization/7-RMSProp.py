#!/usr/bin/env python3
"""
This module provides a function to update a variable using the RMSProp optimization algorithm.
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    Parameters
    ----------
    alpha : float
        The learning rate.
    beta2 : float
        The RMSProp weight.
    epsilon : float
        Small number to avoid division by zero.
    var : np.ndarray
        The variable to be updated.
    grad : np.ndarray
        The gradient of var.
    s : np.ndarray
        The previous second moment of var.

    Returns
    -------
    var : np.ndarray
        The updated variable.
    s : np.ndarray
        The new moment.
    """
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)
    var_new = var - alpha * grad / (np.sqrt(s_new) + epsilon)
    return var_new, s_new
