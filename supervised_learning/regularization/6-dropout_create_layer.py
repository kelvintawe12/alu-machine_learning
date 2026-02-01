#!/usr/bin/env python3
"""
Creates a layer of a neural network using dropout in TensorFlow.
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a layer of a neural network using dropout.

    Parameters
    ----------
    prev : tf.Tensor
        Output of the previous layer.
    n : int
        Number of nodes the new layer should contain.
    activation : callable or None
        Activation function to use on the layer.
    keep_prob : float
        Probability that a node will be kept.

    Returns
    -------
    tf.Tensor
        Output of the new layer.
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dense = tf.layers.Dense(units=n, activation=activation, kernel_initializer=initializer)(prev)
    return tf.layers.Dropout(rate=1 - keep_prob)(dense)
