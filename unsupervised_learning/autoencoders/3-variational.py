#!/usr/bin/env python3
"""
Variational Autoencoder
"""

import tensorflow.keras as keras



def sampling(args):
    """
    Reparameterization trick by sampling from an isotropic unit Gaussian.
    Args:
        z: sampled latent vector
    """
    mu, log_var = args
    batch = tf.shape(mu)[0]
    dim = tf.shape(mu)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return mu + tf.exp(0.5 * log_var) * epsilon



def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes for each hidden layer in the encoder.
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    mu = keras.layers.Dense(latent_dims)(x)
    log_var = keras.layers.Dense(latent_dims)(x)
    z = keras.layers.Lambda(sampling)([mu, log_var])
    encoder = keras.Model(inputs, [z, mu, log_var])

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs)

    # VAE loss
    def vae_loss(y_true, y_pred):
        reconstruction_loss = keras.losses.binary_crossentropy(y_true, y_pred)
        reconstruction_loss *= input_dims
        kl_loss = 1 + log_var - tf.square(mu) - tf.exp(log_var)
        kl_loss = -0.5 * tf.reduce_sum(kl_loss, axis=-1)
        return tf.reduce_mean(
            reconstruction_loss + kl_loss
        )

    # Autoencoder
    auto_inputs = inputs
    z, _, _ = encoder(auto_inputs)
    decoded = decoder(z)
    auto = keras.Model(auto_inputs, decoded)
    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
