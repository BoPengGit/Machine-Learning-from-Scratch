import numpy as np

def binary_cross_entropy(y_predict, y_actual):
    loss = -(y_actual*np.log(y_predict) + (1-y_actual)*np.log(1-y_predict))
    return loss

def categorical_cross_entropy(x, theta_array):
    sigmoid = 1/(1+np.exp(-np.dot(x.transpose(), theta_array)))
    return sigmoid

def rmse(x, theta_array, y):
    # root mean square error
    avg_minibatch_loss = np.sqrt(
             np.average(
             np.square(
             x.transpose().dot(theta_array) - y)))
    return avg_minibatch_loss

def svm_loss(x, theta_array, y):
    pass
