#!/usr/bin/env python3
"""
Deep neural network for multiclass classification (activation function support)
"""
import numpy as np
import pickle
import os


class DeepNeuralNetwork:
    """Deep neural network for multiclass classification (activation function support)"""
    def __init__(self, nx, layers, activation='sig'):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")
        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation
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

    @property
    def activation(self):
        return self.__activation

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
            if l == self.__L:
                # Output layer: softmax for multiclass
                t = np.exp(Zl - np.max(Zl, axis=0, keepdims=True))
                self.__cache['A' + str(l)] = t / np.sum(t, axis=0, keepdims=True)
            else:
                if self.__activation == 'sig':
                    self.__cache['A' + str(l)] = 1 / (1 + np.exp(-Zl))
                else:
                    self.__cache['A' + str(l)] = np.tanh(Zl)
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A + 1e-8)) / m
        return cost

    def evaluate(self, X, Y):
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.zeros_like(A)
        prediction[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        m = Y.shape[1]
        L = self.__L
        weights = self.__weights.copy()
        A_last = cache['A' + str(L)]
        dZ = A_last - Y
        for l in reversed(range(1, L + 1)):
            A_prev = cache['A' + str(l - 1)]
            Wl = weights['W' + str(l)]
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            self.__weights['W' + str(l)] = self.__weights['W' + str(l)] - alpha * dW
            self.__weights['b' + str(l)] = self.__weights['b' + str(l)] - alpha * db
            if l > 1:
                dA_prev = np.matmul(Wl.T, dZ)
                if self.__activation == 'sig':
                    dZ = dA_prev * (A_prev * (1 - A_prev))
                else:
                    dZ = dA_prev * (1 - np.square(A_prev))

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if (verbose or graph):
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        costs = []
        steps = []
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                c = self.cost(Y, A)
                if verbose:
                    print(f"Cost after {i} iterations: {c}")
                if graph:
                    costs.append(c)
                    steps.append(i)
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)
        if graph:
            import matplotlib.pyplot as plt
            plt.plot(steps, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        if not os.path.exists(filename):
            return None
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
