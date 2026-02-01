#!/usr/bin/env python3
"""
Creates placeholders for input data and labels for a neural network in TensorFlow 1.x
"""
import tensorflow as tf

def create_placeholders(nx, classes):
    
    
    """
    Returns two placeholders, x and y, for the neural network
    Args:
        nx (int): number of feature columns in the data
        classes (int): number of classes in the classifier
    Returns:
        x: placeholder for the input data, shape (None, nx)
        y: placeholder for the one-hot labels, shape (None, classes)
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y
