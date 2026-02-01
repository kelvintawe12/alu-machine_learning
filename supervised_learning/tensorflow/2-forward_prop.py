#!/usr/bin/env python3
"""
Creates the forward propagation graph for a neural network in TensorFlow 1.x
"""
import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network

    Args:
        x: placeholder for the input data

        layer_sizes (list): number of nodes in each layer
        activations (list): activation functions for each layer
    Returns:
        The prediction of the network in tensor form
    """
    output = x
    for i in range(len(layer_sizes)):
        activation = activations[i] if i < len(activations) else None
        output = create_layer(output, layer_sizes[i], activation)
    return output
