import numpy as np


class LinearRegression(object):
    """Multivariate linear regression model using batch gradient descent"""

    def __init__(self):
        pass

    def train(self, x, y, epochs, learning_rate):

        self.theta_array = np.zeros(np.array(x.ndim)+1)

        x = self._add_bias(x)

        for _ in range(1, epochs):
            for index, theta in enumerate(self.theta_array):
                avg_full_batch_loss = np.average(np.sqrt(np.square(np.dot(x[index], theta) - y)))
                avg_full_batch_partial_derrivitives = 0.5*np.average((np.dot(x[index], theta)- y)*x[index])
                self.theta_array[index] -= learning_rate * avg_full_batch_partial_derrivitives

    def validate(self, x, y):
        self._check_theta_exists()

        x = self._add_bias(x)

        predicted_y = np.dot(x.transpose(), self.theta_array)
        rmse = np.sqrt(np.average(np.square(y- predicted_y))) # Root Mean Square Error (RMSE)
        return predicted_y, rmse

    def predict(self, x):
        self._check_theta_exists()

        x = self._add_bias(x)

        predicted_y = np.dot(x.transpose(), self.theta_array)
        return predicted_y

    def _add_bias(self, x):
        if x.ndim == 1:
             x = np.row_stack((x, np.ones(len(x))))
        else:
             x = np.row_stack((x, np.ones(len(x[0]))))
        return x

    def _check_theta_exists(self):
        assert hasattr(self, 'theta_array'), ("ValueError: theta is not defined. "
            "Please make sure to train the model before predicting.")

    
