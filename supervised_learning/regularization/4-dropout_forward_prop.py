#!/usr/bin/env python3
"""
Conducts forward propagation using Dropout regularization.
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (nx, m).
    weights : dict
        Weights and biases of the neural network.
    L : int
        Number of layers in the network.
    keep_prob : float
        Probability that a node will be kept.

    Returns
    -------
    dict
        Outputs and dropout masks for each layer.
    """
    np.random.seed(0)
    cache = {'A0': X}
    for l in range(1, L + 1):
        W = weights['W{}'.format(l)]
        b = weights['b{}'.format(l)]
        Z = np.matmul(W, cache['A{}'.format(l - 1)]) + b
        if l != L:
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=A.shape)
            A *= D
            A /= keep_prob
            cache['D{}'.format(l)] = D
        else:
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        cache['A{}'.format(l)] = A
    return cache
