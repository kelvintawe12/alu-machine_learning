#!/usr/bin/env python3
"""
Deep neural network for binary classification (forward propagation)
"""
import numpy as np


class DeepNeuralNetwork:
    """Deep neural network for binary classification (forward propagation)"""
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for l in range(1, self.__L + 1):
            layer_size = layers[l - 1]
            prev_size = nx if l == 1 else layers[l - 2]
            self.__weights['W' + str(l)] = (
                np.random.randn(layer_size, prev_size) * np.sqrt(2 / prev_size)
            )
            self.__weights['b' + str(l)] = np.zeros((layer_size, 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the deep neural network
        Args:
            X (np.ndarray): Input data of shape (nx, m)
        Returns:
            tuple: Activated output (A, cache)
        """
        self.__cache['A0'] = Xy
        for l in range(1, self.__L + 1):
            Wl = self.__weights['W' + str(l)]
            bl = self.__weights['b' + str(l)]
            Al_prev = self.__cache['A' + str(l - 1)]
            Zl = np.matmul(Wl, Al_prev) + bl
            self.__cache['A' + str(l)] = 1 / (1 + np.exp(-Zl))
        return self.__cache['A' + str(self.__L)], self.__cache
