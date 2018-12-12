from mlscratch.activation import *


class FullyConnected(object):

    def __init__(self, size, activation):
        self.size = size
        self.activation = activation()


class Conv2D(object):
    pass


class Flatten(object):
    pass


class InputLayer(object):

    def __init__(self, size):
        self.size = size


class batch_normalization(object):
    pass


class Dropout(object):
    pass
