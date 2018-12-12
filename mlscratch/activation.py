import numpy as np

class no_activation(object):

    def __init__(self):
        pass

class sigmoid(object):

    def __init__(self):
        pass

    def evaluate(self, array):
        self.sigmoid = 1/(1+np.exp(-array))
        return self.sigmoid

    def derivative(self, output_values, input_values):
        return self.sigmoid * (1 - self.sigmoid) * input_values

class tanh(object):

    def __init__(self):
        pass

    def evaluate(self, array):
        pass

    def derivative(self):
        pass

class relu(object):

    def __init__(self):
        pass

    def evaluate(self, array):
        return np.maximum(array, 0)

    def derivative(self, output_values, input_values):
        return np.outer(input_values, (1 * (output_values > 0)))

class leaky_relu(object):

    def __init__(self):
        pass

    def evaluate(self, array):
        pass

    def derivative(self):
        pass

class elu(object):

    def __init__(self):
        pass

    def evaluate(self, array):
        pass

    def derivative(self):
        pass
