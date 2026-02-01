#!/usr/bin/env python3
"""
This module provides a function to
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow.

    Parameters
    ----------
    
    Returns
    -------
    tf.Tensor
        The activated output for the layer.
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dense = tf.layers.Dense(units=n, kernel_initializer=initializer)(prev)
    mean, variance = tf.nn.moments(dense, axes=[0])
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    epsilon = 1e-8
    batch_norm = tf.nn.batch_normalization(dense, mean, variance, beta, gamma, epsilon)
    if activation is not None:
        return activation(batch_norm)
    return batch_norm
