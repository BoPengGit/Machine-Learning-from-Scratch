import numpy as np


class LinearRegression(object):
    """Multvariate Logistic regression model using batch gradient descent"""

    def __init__(self):
        pass

    def train(self, x, y, epochs, learning_rate):
        self.theta_array = np.zeros(x.shape[0])

        for _ in range(0, epochs):
            for index, theta in enumerate(self.theta_array):
                sigmoid_loss = - np.log(h(x))
                avg_full_batch_loss = 0.5*np.average(np.square(np.dot(x[index], theta) - y))
                avg_full_batch_partial_derrivitives = np.average(np.dot(x[index], theta)- y)
                self.theta_array[index] = theta - learning_rate * avg_full_batch_partial_derrivitives

    def validate(self, x, y):
        # assert 'self.theta_array' in globals(), ("ValueError: theta is not defined."
        #     "Please make sure to train the model before validating.")

        predicted_y = np.dot(x, self.theta_array)
        accuracy = len(y) - np.sum(np.absolute(y_predicted - y))
        return predicted_y, accuracy

    def predict(self, x):
        # assert 'self.theta_array' in globals(), ("ValueError: theta is not defined."
        #     "Please make sure to train the model before predicting.")
        predicted_y = np.dot(self.theta_array, x)
        return predicted_y
