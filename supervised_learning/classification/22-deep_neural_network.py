#!/usr/bin/env python3
"""
Deep neural network for binary classification (train method)
"""
import numpy as np


class DeepNeuralNetwork:
    """Deep neural network for binary classification (train method)"""
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
        self.__cache['A0'] = X
        for l in range(1, self.__L + 1):
            Wl = self.__weights['W' + str(l)]
            bl = self.__weights['b' + str(l)]
            Al_prev = self.__cache['A' + str(l - 1)]
            Zl = np.matmul(Wl, Al_prev) + bl
            self.__cache['A' + str(l)] = 1 / (1 + np.exp(-Zl))
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        m = Y.shape[1]
        L = self.__L
        weights = self.__weights.copy()
        dZ = cache['A' + str(L)] - Y
        for l in reversed(range(1, L + 1)):
            A_prev = cache['A' + str(l - 1)]
            Wl = weights['W' + str(l)]
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            self.__weights['W' + str(l)] = self.__weights['W' + str(l)] - alpha * dW
            self.__weights['b' + str(l)] = self.__weights['b' + str(l)] - alpha * db
            if l > 1:
                A_prev = cache['A' + str(l - 1)]
                dA_prev = np.matmul(Wl.T, dZ)
                dZ = dA_prev * (A_prev * (1 - A_prev))

    def train(self, X, Y, iterations=5000, alpha=0.05):
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
        return self.evaluate(X, Y)
