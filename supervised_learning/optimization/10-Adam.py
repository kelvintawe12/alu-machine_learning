#!/usr/bin/env python3
"""
This module provides a function to create the training operation for a neural network in tensorflow using the Adam optimization algorithm.
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training operation for a neural network in tensorflow using the Adam optimization algorithm.

    Parameters
    ----------
    loss : tf.Tensor
        The loss of the network.
    
    Returns
    -------
    tf.Operation
        The Adam optimization operation.
    """
    return tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1, beta2=beta2, epsilon=epsilon).minimize(loss)
