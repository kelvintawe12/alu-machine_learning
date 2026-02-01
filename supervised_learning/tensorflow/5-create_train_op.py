#!/usr/bin/env python3
"""
Creates the training operation for a neural network in TensorFlow 1.x
"""
import tensorflow as tf

def create_train_op(loss, alpha):
    """
    Creates the training operation for the network
    Args:
        loss: loss of the network’s prediction
        alpha: learning rate
    Returns:
        An operation that trains the network using gradient descent
    """
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = optimizer.minimize(loss)
    return train_op
