import numpy as np


class LinearRegression(object):
    """ Initially this is a linear regression with theta1 and theta0 as parameters and using
        batch gradient descent.
    """

    def __init__(self):
        self.theta1 = 0

    def train(self, x, y, epochs, learning_rate):
        for _ in range(0, epochs):
            average_full_batch_loss = 0.5*avg((np.dot(x, theta) - y)**2)
            average_full_batch_partial_derrivitives = avg(np.dot(x, theta)- y)
            theta = theta + learning_rate * average_full_batch_partial_derriviteis
        self.theta = theta

    def validate(self, x, y):
        predicted_y = np.dot(x, self.theta)
        accuracy_score = sqrt(avg((y- predicted_y)**2)) # Root Mean Square Error (RMSE)
        return predicted_y, accuracy_score

    def predict(self, x):
        predicted_y = np.dot(x, self.theta)
        return predicted_y
