#!/usr/bin/env python3
import tensorflow.keras as keras

"""
Vanilla Autoencoder
This module defines a function to create a vanilla autoencoder using Keras.
"""

import tensorflow.keras as keras

def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Vanilla Autoencoder
    This module defines a function to create a vanilla autoencoder using Keras.
    """


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder.

    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes for each hidden layer in the encoder.
        latent_dims (int): Dimensions of the latent space representation.

    Returns:
        encoder: The encoder model.
        decoder: The decoder model.
        auto: The full autoencoder model.
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    latent = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(inputs, latent)

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(
        input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs)

    # Autoencoder
    auto_inputs = inputs
    encoded = encoder(auto_inputs)
    decoded = decoder(encoded)
    auto = keras.Model(auto_inputs, decoded)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
