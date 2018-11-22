import numpy as np

def adam(theta_array,
         learning_rate,
         beta1,
         beta2,
         velocity,
         sqr_velocity,
         epsilon,
         gradient):
    """Adaptive moment estimation (Adam)"""
    new_velocity = beta1 * velocity + (1- beta1) * gradient
    new_sqr_velocity = beta2 * sqr_velocity + (1 - beta2) * gradient**2
    theta_array -= learning_rate * new_velocity / np.sqrt(new_sqr_velocity + epsilon)
    return theta_array, velocity, sqr_velocity

def gradient_descent(theta_array, learning_rate, gradient):
    """Vanilla Gradient Descent. Can be batch, minibatch, or stocastic."""
    theta_array -= learning_rate * gradient
    return theta_array

def momentum(theta_array, learning_rate, beta, velocity, gradient):
    """Gradient Descent with momentum vector"""

    new_velocity = beta * velocity + (1- beta) * gradient
    theta_array -= learning_rate * new_velocity
    return theta_array, new_velocity

def rmsprop(theta_array, learning_rate, beta, sqr_velocity, epsilon, gradient):

    new_sqr_velocity = beta * sqr_velocity + (1 - beta) * gradient**2
    theta_array -= learning_rate * gradient / np.sqrt(new_sqr_velocity + epsilon)
    return theta_array, sqr_velocity
