#!/usr/bin/env python3
"""
Calculates the cost of a neural network with L2 regularization in TensorFlow.
"""
import tensorflow as tf


def l2_reg_cost(cost):
    """
    Calculates the cost of a neural network with L2 regularization in TensorFlow.

    Parameters
    ----------
    cost : tf.Tensor
        The cost of the network without L2 regularization.

    Returns
    -------
    tf.Tensor
        The cost of the network accounting for L2 regularization.
    """
    return cost + tf.losses.get_regularization_loss()
