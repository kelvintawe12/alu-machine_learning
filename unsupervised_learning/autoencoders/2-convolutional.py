#!/usr/bin/env python3
"""
Convolutional Autoencoder
This module defines a function to create a convolutional autoencoder using Keras.
"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Args:
        input_dims (tuple): Dimensions of the model input.
        filters (list): Number of filters for each conv layer in the encoder.
        latent_dims (tuple): Dimensions of the latent space representation.

    Returns:
        encoder: The encoder model.
        decoder: The decoder model.
        auto: The full autoencoder model.
    """
    # Encoder
    inputs = keras.Input(shape=input_dims)
    x = inputs
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    shape_before_flatten = keras.backend.int_shape(x)[1:]
    x = keras.layers.Flatten()(x)
    latent = keras.layers.Dense(
        int(keras.backend.prod(latent_dims)), activation='relu')(x)
    latent_reshaped = keras.layers.Reshape(latent_dims)(latent)
    encoder = keras.Model(inputs, latent_reshaped)

    # Decoder
    latent_inputs = keras.Input(shape=latent_dims)
    x = keras.layers.Flatten()(latent_inputs)
    x = keras.layers.Dense(
        keras.backend.prod(shape_before_flatten), activation='relu')(x)
    x = keras.layers.Reshape(shape_before_flatten)(x)
    for i, f in enumerate(reversed(filters)):
        if i < len(filters) - 2:
            x = keras.layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
            x = keras.layers.UpSampling2D((2, 2))(x)
        elif i == len(filters) - 2:
            x = keras.layers.Conv2D(f, (3, 3), activation='relu', padding='valid')(x)
            x = keras.layers.UpSampling2D((2, 2))(x)
        else:
            x = keras.layers.Conv2D(
                input_dims[-1], (3, 3), activation='sigmoid', padding='same')(x)
    decoder = keras.Model(latent_inputs, x)

    # Autoencoder
    auto_inputs = inputs
    encoded = encoder(auto_inputs)
    decoded = decoder(encoded)
    auto = keras.Model(auto_inputs, decoded)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
