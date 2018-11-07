import numpy as np

def gradient_descent(theta_array, learning_rate, gradient):
    """Vanilla Gradient Descent. Can be batch, minibatch, or stocastic."""

    return theta_array - learning_rate * gradient

def adam(learning_rate, moment, gradient):
    pass

def momentum(learning_rate, momentum, gradient):
    pass

def rmsprop(learning_rate, beta, gradient):
    pass
