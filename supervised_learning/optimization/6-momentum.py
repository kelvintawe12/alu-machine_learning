#!/usr/bin/env python3
"""
This module provides a function to create the training operation for a neural network in tensorflow using the gradient descent with momentum optimization algorithm.
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Creates the training operation for a neural network in tensorflow using the gradient descent with momentum optimization algorithm.

    Parameters
    ----------
    loss : tf.Tensor
        The loss of the network.
    alpha : float
        The learning rate.
    beta1 : float
        The momentum weight.

    Returns
    -------
    tf.Operation
        The momentum optimization operation.
    """
    return tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1).minimize(loss)
