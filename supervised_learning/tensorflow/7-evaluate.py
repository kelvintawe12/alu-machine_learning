#!/usr/bin/env python3
"""
Evaluates the output of a neural network in TensorFlow 1.x
"""
import tensorflow as tf
import numpy as np

def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network
    Args:
        X: numpy.ndarray, input data to evaluate
        Y: numpy.ndarray, one-hot labels for X
        save_path: str, location to load the model from
    Returns:
        The network’s prediction, accuracy, and loss, respectively
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        y_pred_val, acc_val, loss_val = sess.run(
            [y_pred, accuracy, loss], feed_dict={x: X, y: Y})
        return y_pred_val, acc_val, loss_val
