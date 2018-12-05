import numpy as np


class sigmoid(object):

    def __init__(self, array):
        self.array = array

    def __new__(cls):
        sigmoid = 1/(1+np.exp(array))
        return sigmoid

    def derivative(self):
        pass

class tanh(object):

    def __init__(self, array):
        self.array = array

    def __new__(cls):
        pass

    def derivative(self):
        pass

class relu(object):

    def __init__(self, array):
        self.array = array

    def __new__(cls):
        return np.maximum(array, 0, array)

    def derivative(self):
        pass

class leaky_relu(object):

    def __init__(self, array):
        self.array = array

    def __new__(cls):
        pass

    def derivative(self):
        pass

class elu(object):

    def __init__(self, array):
        self.array = array

    def __new__(cls):
        pass

    def derivative(self):
        pass
