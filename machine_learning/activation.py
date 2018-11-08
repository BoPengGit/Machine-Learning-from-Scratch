import numpy as np


def sigmoid(x, theta_array):
    sigmoid = 1/(1+np.exp(-np.dot(x.transpose(), theta_array)))
    return sigmoid

def tanh(x, theta_array):
    pass

def relu(x, theta_array):
    pass

def leaky_relu(x, theta_array):
    pass
    
def elu(x, theta_array):
    pass
