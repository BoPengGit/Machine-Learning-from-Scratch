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
        self.architecture = []
        self.weights = []
        self.local_gradients = []
        self.partial_loss_gradients = []
        self.loss_function = None
        self.optimizer = None

    def add(self, layer):
        self.architecture.append(layer)

    def compile(self):
        for i in range(1, len(self.architecture)):
            self.weights.append(self._weight_initialization(self.architecture[i-1].size+1,
                                                            self.architecture[i].size))

    def validate(self, x_data, y):
        pass

    def predict(self, x_data):
        return self._forward_pass(x_data)

    def train(self, x_data, y, batch_size):
        pass

    def _forward_pass(self, x_data):
        y_predict = []
        for x in x_data:
            input_values = x
            for index, weights_layer in enumerate(self.weights):
                input_values = np.append(input_values, 1)

                output_values = self.architecture[index+1].activation.evaluate((np.dot(input_values,
                                                                                       weights_layer.transpose())))
                self.local_gradients.append(self.architecture[index+1].activation.derivative(output_values,
                                                                                             input_values))
                input_values = output_values
            y_predict_x = input_values
            y_predict.append(y_predict_x)

        return y_predict

    def _backward_pass(self):


    def _weight_initialization(self, size_l_minus1, size_l):
        weight_matrix = (np.random.randn(size_l, size_l_minus1) * np.sqrt(2/size_l_minus1)) * 0.01 + 0.05
        return weight_matrix
