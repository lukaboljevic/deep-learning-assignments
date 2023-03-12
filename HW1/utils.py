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


def cross_entropy(y_true, y_pred, epsilon=1e-12, lmbd=0.0, n=None, weights=None):
    """
    Calculate cross entropy, given predictions and ground truths.

    Parameters:
        y_true  - ground truths i.e. targets for the provided predictions

        y_pred  - predictions, i.e. the output of the neural network

        epsilon - small constant for numerical stability

        lmbd    - L2 regularization parameter

        n       - size of the TODO, used when calculating the regularization term

        weights - weights of the neural network
    """
    targets = y_true.transpose()
    predictions = y_pred.transpose()
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    num_samples = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / num_samples

    if lmbd != 0.0:
        if not weights:
            raise ValueError("No weights provided for calculating L2 regularization term.")
        if not n:
            raise ValueError("Size of training data (n) not given for calculating L2 regularization term.")

        # L2 regularization
        sum = 0
        for layer_weights in weights:
            sum += np.sum(np.square(layer_weights))
        reg_term = (lmbd / 2*n) * sum
        ce += reg_term
    return ce


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')
    

def load_data_cifar(train_file, test_file):
    """
    Load the CIFAR10 dataset
    """
    train_dict = unpickle(train_file)
    test_dict = unpickle(test_file)
    train_data = np.array(train_dict['data']) / 255.0
    train_class = np.array(train_dict['labels'])
    train_class_one_hot = np.zeros((train_data.shape[0], 10))
    train_class_one_hot[np.arange(train_class.shape[0]), train_class] = 1.0
    test_data = np.array(test_dict['data']) / 255.0
    test_class = np.array(test_dict['labels'])
    test_class_one_hot = np.zeros((test_class.shape[0], 10))
    test_class_one_hot[np.arange(test_class.shape[0]), test_class] = 1.0
    
    return train_data.transpose(), train_class_one_hot.transpose(), test_data.transpose(), test_class_one_hot.transpose()
