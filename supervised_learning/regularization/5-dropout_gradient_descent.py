#!/usr/bin/env python3
"""
Updates the weights of a neural network with Dropout regularization using gradient descent.
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization using gradient descent.

    Parameters
    ----------
    Y : np.ndarray
        One-hot array of shape (classes, m) with correct labels.
    weights : dict
        Weights and biases of the neural network.
    cache : dict
        Outputs and dropout masks of each layer.
    alpha : float
        Learning rate.
    keep_prob : float
        Probability that a node will be kept.
    L : int
        Number of layers of the network.
    """
    m = Y.shape[1]
    dZ = cache['A{}'.format(L)] - Y
    for l in range(L, 0, -1):
        A_prev = cache['A{}'.format(l - 1)]
        W = weights['W{}'.format(l)]
        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        weights['W{}'.format(l)] -= alpha * dW
        weights['b{}'.format(l)] -= alpha * db
        if l > 1:
            dA_prev = np.matmul(W.T, dZ)
            dA_prev *= cache['D{}'.format(l - 1)]
            dA_prev /= keep_prob
            dZ = dA_prev * (1 - cache['A{}'.format(l - 1)] ** 2)
