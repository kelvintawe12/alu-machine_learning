#!/usr/bin/env python3
"""
Calculates the accuracy of a prediction in TensorFlow 1.x
"""
import tensorflow as tf

def calculate_accuracy(y, y_pred):
    
    
    """
    Calculates the accuracy of a prediction
    Args:
        y: placeholder for the labels of the input data
        y_pred: tensor containing the network’s predictions
    Returns:
        A tensor containing the decimal accuracy of the prediction
    """
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy
