#!/usr/bin/env python3
"""
This module provides a function to create the training operation for a neural network in tensorflow using the RMSProp optimization algorithm.
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates the training operation for a neural network in tensorflow using the RMSProp optimization algorithm.

    Parameters
    ----------
    loss : tf.Tensor
        The loss of the network.
    alpha : float
        The learning rate.
    beta2 : float
        The RMSProp weight.
    epsilon : float
        Small number to avoid division by zero.

    Returns
    -------
    tf.Operation
        The RMSProp optimization operation.
    """
    return tf.train.RMSPropOptimizer(learning_rate=alpha, decay=beta2, epsilon=epsilon).minimize(loss)
