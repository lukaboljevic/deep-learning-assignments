import numpy as np
import pickle


def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=0, keepdims=True)


def softmax_dLdZ(output, target):
    """
    Partial derivative of the cross entropy loss w.r.t Z at the last layer
    """
    return output - target


def cross_entropy(y_true, y_pred, epsilon=1e-12):
    targets = y_true.transpose()
    predictions = y_pred.transpose()
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')