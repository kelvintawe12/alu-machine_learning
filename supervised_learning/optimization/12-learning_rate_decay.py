#!/usr/bin/env python3
"""
This module provides a function
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates a learning rate decay operation in tensorflow using inverse time decay.


    tf.Tensor
        The learning rate decay operation.
    """
    return tf.train.inverse_time_decay(alpha, global_step, decay_step, decay_rate, staircase=True)
