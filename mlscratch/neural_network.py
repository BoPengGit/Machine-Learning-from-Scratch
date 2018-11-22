import numpy as np


class NeuralNetwork(object):
    """Neural Network class.
        Methods:
            __init__(self):
                initialize the specific (self) NeuralNetwork object.
            add(self, layer):
                Add a layer to the NeuralNetwork (self) object.
            compile(self):
                Compile the NeuralNetwork (self) object to be ready to be
                trained/valided/predicted. Connects the forward and backward pass
                math of the (self) NeuralNetwork object. initialize the weights
                of the (self) NeuralNetwork object.
            train(self, x, y, batch_size, epoch, optimizer, learning_rate,
                  activation_function, cost_function):
                Updates the weights of the compiled architecture of the (self)
                neural network object through forward and backward pass.
                    Parameters:
                        x: The input data.
                        y: The target data values.
                        batch_size: Number of training examples used to update
                        one backward pass.
                            Example:
                                The batch_size (int) number of examples will go
                                through the forward pass to compute the loss
                                of each example. The average loss of all of the
                                examples is used as the loss to calculate the loss
                                gradient and update the weights through the
                                backward pass.
                        epoch: Number of times to do forward and backward pass
                        through the entire training data (x).
                        optimizer: The optimization function used to update
                        the weights during the backward pass.
                        learning_rate: Learing_rate used to update the weights
                        during the backward pass.
                        activation_function:
                        cost_function: Function to calculate the loss.
            validate(self, x, y):
                Predicts target y values of x input using the current weights
                of the (self) NeuralNetwork object.
                    Output:
                        y_hat: Predicted y values using the current weights
                        of the (self) NeuralNetwork object.
    """

    def __init__(self):
        pass

    def add(self, layer):
        pass

    def compile(self):
        pass

    def train(self, x, y):
        pass

    def validate(self, x, y):
        pass

    def predict(self, x):
        pass

    def _forward_pass(self):
        pass

    def _backword_pass(self):
        pass




# initliaze
# add layers to self object
# compile
# train
# validate / predict
