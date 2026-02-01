#!/usr/bin/env python3
"""
This module provides a function to train a loaded neural network model using mini-batch gradient descent.
"""
import numpy as np
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data

def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch gradient descent.

    Parameters
    ----------
    X_train : np.ndarray (m, 784)
        Training data.
    Y_train : np.ndarray (m, 10)
        Training labels (one-hot).
    X_valid : np.ndarray (m, 784)
        Validation data.
    Y_valid : np.ndarray (m, 10)
        Validation labels (one-hot).
    batch_size : int
        Number of data points in a batch.
    epochs : int
        Number of epochs to train.
    load_path : str
        Path to load the model from.
    save_path : str
        Path to save the model to.

    Returns
    -------
    str
        The path where the model was saved.
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]
        m = X_train.shape[0]
        steps_per_epoch = int(np.ceil(m / batch_size))
        for epoch in range(epochs + 1):
            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            train_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            valid_acc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_acc))
            if epoch == epochs:
                break
            X_shuff, Y_shuff = shuffle_data(X_train, Y_train)
            for step in range(steps_per_epoch):
                start = step * batch_size
                end = start + batch_size
                X_batch = X_shuff[start:end]
                Y_batch = Y_shuff[start:end]
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                if (step + 1) % 100 == 0 and step != 0:
                    step_cost = sess.run(loss, feed_dict={x: X_batch, y: Y_batch})
                    step_acc = sess.run(accuracy, feed_dict={x: X_batch, y: Y_batch})
                    print("\tStep {}:".format(step + 1))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_acc))
        return saver.save(sess, save_path)
