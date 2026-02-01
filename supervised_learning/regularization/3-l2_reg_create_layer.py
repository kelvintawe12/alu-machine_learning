#!/usr/bin/env python3
"""
Creates a tensorflow layer that includes L2 regularization.
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a tensorflow layer that includes L2 regularization.

    Parameters
    ----------
    prev : tf.Tensor
        Output of the previous layer.
    n : int
        Number of nodes the new layer should contain.
    activation : callable or None
        Activation function to use on the layer.
    lambtha : float
        L2 regularization parameter.

    Returns
    -------
    tf.Tensor
        Output of the new layer.
    """
    kernel_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    kernel_reg = tf.contrib.layers.l2_regularizer(lambtha)
    return tf.layers.Dense(units=n, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(prev)
