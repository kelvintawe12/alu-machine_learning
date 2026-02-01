#!/usr/bin/env python3
"""
This module provides a function to update a variable using the gradient descent with momentum optimization algorithm.
"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with momentum optimization algorithm.

    Parameters
    ----------
    alpha : float
        The learning rate.
    beta1 : float
        The momentum weight.
    var : np.ndarray
        The variable to be updated.
    grad : np.ndarray
        The gradient of var.
    v : np.ndarray
        The previous first moment of var.

    Returns
    -------
    var : np.ndarray
        The updated variable.
    v : np.ndarray
        The new moment.
    """
    v_new = beta1 * v + (1 - beta1) * grad
    var_new = var - alpha * v_new
    return var_new, v_new
