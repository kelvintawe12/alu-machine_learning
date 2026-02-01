#!/usr/bin/env python3
"""
Updates the weights and biases of a neural network using gradient descent with L2 regularization.
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent with L2 regularization.

    Parameters
    ----------
    Y : np.ndarray
        One-hot array of shape (classes, m) with correct labels.
    weights : dict
        Weights and biases of the neural network.
    cache : dict
        Outputs of each layer of the neural network.
    alpha : float
        Learning rate.
    lambtha : float
        L2 regularization parameter.
    L : int
        Number of layers of the network.
    """
    m = Y.shape[1]
    dZ = cache['A{}'.format(L)] - Y
    for l in range(L, 0, -1):
        A_prev = cache['A{}'.format(l - 1)]
        W = weights['W{}'.format(l)]
        dW = (np.matmul(dZ, A_prev.T) / m) + (lambtha / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m
        weights['W{}'.format(l)] -= alpha * dW
        weights['b{}'.format(l)] -= alpha * db
        if l > 1:
            dA_prev = np.matmul(W.T, dZ)
            dZ = dA_prev * (1 - cache['A{}'.format(l - 1)] ** 2)
