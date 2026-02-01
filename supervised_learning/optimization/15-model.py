#!/usr/bin/env python3
"""
This module provides a function to build.
"""
import numpy as np
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data
create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer

def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in tensorflow using Adam optimization, mini-batch gradient descent, learning rate decay, and batch normalization.

    Returns
    -------
    str
        Path where the model was saved.
    """
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    nx = X_train.shape[1]
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, Y_train.shape[1]), name='y')
    a = x
    for i in range(len(layers)):
        act = activations[i]
        if i != len(layers) - 1:
            a = create_batch_norm_layer(a, layers[i], act)
        else:
            initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
            a = tf.layers.Dense(units=layers[i], activation=act, kernel_initializer=initializer)(a)
    y_pred = a
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1)), tf.float32))
    global_step = tf.Variable(0, trainable=False)
    alpha_decay = tf.train.inverse_time_decay(alpha, global_step, 1, decay_rate, staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate=alpha_decay, beta1=beta1, beta2=beta2, epsilon=epsilon).minimize(loss, global_step=global_step)
    saver = tf.train.Saver()
    m = X_train.shape[0]
    steps_per_epoch = int(np.ceil(m / batch_size))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
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
