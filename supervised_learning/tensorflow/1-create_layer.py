#!/usr/bin/env python3
"""
Creates a layer for a neural network using TensorFlow 1.x
"""
import tensorflow as tf

def create_layer(prev, n, activation):
    """
    Creates a layer for the neural network
    Args:
        prev: tensor output of the previous layer
        n (int): number of nodes in the layer to create
        activation: activation function for the layer
    Returns:
        The tensor output of the layer
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name="layer"
    )
    return layer(prev)
