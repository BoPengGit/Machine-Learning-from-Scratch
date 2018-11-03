import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline


class LinearRegression(object):
    """ Linear regression model using batch gradient descent"""

    def __init__(self):
        pass

    def train(self, x, y, epochs, learning_rate):
        self.theta_array = np.zeros(x.shape[1])

        for _ in range(0, epochs):
            for index, theta in enumerate(self.theta_array):
                average_full_batch_loss = 0.5*np.average((np.dot(x[index], np.array(theta).transpose) - y)**2)
                average_full_batch_partial_derrivitives = np.average(np.dot(x[index], np.array(theta).transpose)- y)
                self.theta_array[index] = theta - learning_rate * average_full_batch_partial_derrivitives

    def validate(self, x, y):
#         assert 'self.theta_array' in globals(), ("ValueError: theta is not defined. Please make sure to train the model "
#             "before validating.")

        predicted_y = np.dot(x, self.theta_array.transpose)
        accuracy_score = np.sqrt(np.average((y- predicted_y)**2)) # Root Mean Square Error (RMSE)
        return predicted_y, accuracy_score

    def predict(self, x):
#         assert 'self.theta_array' in globals(), ("ValueError: theta is not defined. Please make sure to train the model "
#             "before predicting.")

        predicted_y = np.dot(x, self.theta_array.transpose)
        return predicted_y
