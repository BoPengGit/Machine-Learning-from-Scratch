import numpy as np

def gradient_descent(theta_array, learning_rate, gradient):
    """Vanilla Gradient Descent. Can be batch, minibatch, or stocastic."""

    return theta_array - learning_rate * gradient

def adam(learning_rate, moment, gradient):
    pass

def momentum(theta_array, learning_rate, rho, velocity, gradient):
    """Gradient Descent with momentum vector"""

    new_velocity = rho * velocity + gradient
    return theta_array - learning_rate * new_velocity

def rmsprop(learning_rate, beta, gradient):
    pass
