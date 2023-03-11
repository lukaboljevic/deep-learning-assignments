import matplotlib.pyplot as plt
import numpy as np

from utils import *

EXP_LR = "exponential"


class Network(object):
    """
    Column vectors are assumed - when performing eg. forward pass, the pre-activations (Zs) and 
    activations (As) for single sample are found in a single column.

    Notations taken from lecture slides, i.e. from http://neuralnetworksanddeeplearning.com/
    """

    def __init__(self, sizes, optimizer="sgd"):
        """
        Initialize the weights and biases for the neural network, and optionally the buffers for Adam.

        Parameters:
            sizes - a list of size num_layers, with each element representing the number of neurons 
                    at the respective layer. The first element of the list is the number of input 
                    neurons, while the last element represents the number of output neurons. The 
                    number of inputs must be the same as the number of features, while the number 
                    of outputs must be equal to the number of classes/labels.

        Further info:
            Layers are "numbered" 0 through num_layers - 1. 
            
            self.weights is an array of length num_layers-1. self.weights[0], i.e. weights for layer
            0 correspond to the weights between the input layer (layer 0) and first hidden layer 
            (layer 1). In general, weights for layer L correspond to weights between layer L and L+1.
            Since every neuron in layer L is connected to every neuron in layer L+1 => the weights for
            layer L is of shape [n_L+1, n_L], n_L meaning number of neurons at layer L.

            self.biases is an array of length num_layers-1. Bias for layer L is connected to each neuron 
            in layer L+1, meaning the bias for layer L is of shape [n_L+1, 1].
        """

        # The weights are initialized with a He initializer: https://arxiv.org/pdf/1502.01852v1.pdf
        self.weights = [((2/sizes[i-1])**0.5)*np.random.randn(sizes[i], sizes[i-1]) for i in range(1, len(sizes))]
        self.biases = [np.zeros((x, 1)) for x in sizes[1:]]
        self.optimizer = optimizer
        if self.optimizer == "adam":
            # TODO Implement the buffers necessary for the Adam optimizer.
            pass

    
    def forward_pass(self, input):
        """
        Perform the forward pass.

        Parameters:
            input - numpy array with shape [n0 x m], containing m training samples in the current 
                    mini batch, where n0 is the number of input attributes. m could also be equal
                    to 1, in which case the function was invoked from self.eval_network (i.e. in
                    that case, the input is a single test or validation sample)

        Returns:
            output - array of shape [10, m], containing the output of the network. 10 represents
                     the number of classes, i.e. output neurons,

            Zs     - array of length num_layers-1, containing arrays of shape [n_L, m]. Each element
                     of Zs thus contains the pre-activations for all n_L neurons on layer L, for each
                     of the m elements. There are no pre-activations for the input layer.

            As     - array of length num_layers, containing arrays of shape [n_L, m]. Each element
                     of As thus contains the activations for all n_L neurons on layer L, for each
                     of the m elements.
        """
        # TODO however, the samples coming from self.eval_network are of shape (3072, 1), whereas
        # if they're coming from self.train, they're of shape (3072,) so I will see what
        # suits me more.

        Zs = []
        As = [input]
        num_weights = len(self.weights)
        for l in range(num_weights):
            w = self.weights[l]
            b = self.biases[l]
            z = np.dot(w, As[-1]) + b
            Zs.append(z)

            if l == num_weights - 1:  # weights between final hidden layer and output layer
                As.append(softmax(z))
            else:
                As.append(sigmoid(z))
        
        return As[-1], Zs, As
            

    def backward_pass(self, output, target, Zs, As):
        """
        Perform the backward pass.

        Parameters:
            output - array of shape [10, m], containing the output of the network calculated during 
                     forward pass; m represents the size of the current mini-batch

            target - array of shape [10, m], containing the target values/classes for this batch

            Zs     - array of length num_layers-1, containing arrays of shape [n_L, m]. Each element
                     of Zs thus contains the pre-activations for all n_L neurons on layer L, for each
                     of the m elements. There are no pre-activations for the input layer. 
                     Pre-activations are calculated during forward pass.

            As     - array of length num_layers, containing arrays of shape [n_L, m]. Each element
                     of As thus contains the activations for all n_L neurons on layer L, for each
                     of the m elements. Activations are calculated during forward pass.

        Returns:
            gradients_w - array of length num_layers-1 (same as self.weights), containing arrays of
                          shape [n_L+1, n_L] - same shape as weights for layer L in self.weights.
                          Each element of gradients_w contains the gradient of the loss function
                          w.r.t. the weights (i.e. the partial derivative of the loss function
                          w.r.t. all of the individual weights).

            gradients_b - array of length num_layers-1 (same as self.biases), containing arrays of
                          shape [n_L+1, 1] - same shape as biases for layer L in self.biases.
                          Each element of gradients_b contains the gradient of the loss function
                          w.r.t. the biases (i.e. the partial derivative of the loss function
                          w.r.t. all of the individual biases).

        Further info:
            The indexing used here is not exactly the same as the indexing used in lecture slides, 
            i.e. http://neuralnetworksanddeeplearning.com/. There are a few differences, but to
            avoid confusion, I won't explain it here. It's enough to say that these differences
            in indexing were kept in mind and triple-checked to make sure the calculations are
            correct (and they 100% are).
        """
        batch_size = target.shape[1]
        gradients_w = [np.zeros(w.shape) for w in self.weights]
        gradients_b = [np.zeros(b.shape) for b in self.biases]

        # Backpropagation of the error through the layers
        num_weights = len(self.weights)
        for l in reversed(range(num_weights)):
            if l == num_weights - 1:  # final layer/output error
                delta = softmax_dLdZ(output, target)
            else:
                delta = np.dot(np.transpose(self.weights[l+1]), delta) * sigmoid_prime(Zs[l])
                
            gradients_w[l] = np.dot(delta, np.transpose(As[l])) / batch_size
            gradients_b[l] = np.sum(delta, axis=1, keepdims=True) / batch_size  # not axis=0!
        
        return gradients_w, gradients_b


    def update_network(self, gradients_w, gradients_b, eta):
        """
        Update the weights and biases based on the used optimizer (SGD or Adam).

        Parameters:
            gradients_w - array of length num_layers-1 (same as self.weights), containing arrays of
                          shape [n_L+1, n_L] - same shape as weights for layer L in self.weights.
                          Each element of gradients_w contains the gradient of the loss function
                          w.r.t. the weights (i.e. the partial derivative of the loss function
                          w.r.t. all of the individual weights). These gradients are calculated
                          during backward pass.

            gradients_b - array of length num_layers-1 (same as self.biases), containing arrays of
                          shape [n_L+1, 1] - same shape as biases for layer L in self.biases.
                          Each element of gradients_b contains the gradient of the loss function
                          w.r.t. the biases (i.e. the partial derivative of the loss function
                          w.r.t. all of the individual biases). These gradients are calculated
                          during backward pass.

            eta         - (current) learning rate
        """

        # SGD
        if self.optimizer == "sgd":
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] - eta * gradients_w[i]
                self.biases[i] = self.biases[i] - eta * gradients_b[i]
        elif self.optimizer == "adam":
            # TODO Implement the update function for Adam:
            pass
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")


    def train(self,
              training_data, training_class,
              val_data, val_class,
              epochs,
              mini_batch_size,
              eta, lr_schedule="no",
              **kwargs):
        """
        Train the neural network.

        Parameters:
            training_data   - numpy array of dimensions [n0 x m], where m is the number of training 
                              samples, and n0 is the number of input attributes
            
            training_class  - numpy array of dimensions [c x m], where c is the number of classes

            val_data        - validation data of similar shape to training_data - depends on the
                              percentage of original training data used for validation

            val_class       - validation data classes

            epochs          - number of passes over the dataset

            mini_batch_size - number of examples the network uses to compute the gradient estimation

            eta             - learning rate

        Returns:
            epoch_losses   - a list containing the (average) loss for each epoch

            validation_CAs - a list containing the classification accuracy for each time the network
                             was tested on the validation set
        """

        iteration_index = 0
        eta_current = eta
        n = training_data.shape[1]
        epoch_losses = []
        validation_CAs = []
        if lr_schedule == EXP_LR and not kwargs.get("k"):
            raise ValueError(f"Decay rate k for exponential learning rate decay not provided.")
        k = kwargs.get("k")

        for epoch in range(epochs):
            print()
            print("=" * 60)
            print(f"Epoch {epoch + 1}")
            loss_avg = 0.0
            mini_batches = [
                (training_data[:, k:k + mini_batch_size], training_class[:, k:k + mini_batch_size])
                for k in range(0, n, mini_batch_size)]
            num = len(mini_batches)

            for i, mini_batch in enumerate(mini_batches):
                if (i+1) % 50 == 0: 
                    print(f"\tMini-batch {i+1} / {num}")

                output, Zs, As = self.forward_pass(mini_batch[0])
                gradients_w, gradients_b = \
                    self.backward_pass(output, mini_batch[1], Zs, As)

                self.update_network(gradients_w, gradients_b, eta_current)

                if lr_schedule == "no":
                    eta_current = eta
                elif lr_schedule == EXP_LR:
                    eta_current = eta * np.exp(-k * iteration_index)

                iteration_index += 1

                loss = cross_entropy(mini_batch[1], output)
                loss_avg += loss

            epoch_loss = loss_avg / len(mini_batches)
            epoch_losses.append(epoch_loss)
            print(f"Epoch {epoch + 1} complete")
            print(f"Loss: {epoch_loss}")
            if epoch == 0 or (epoch + 1) % 5 == 0:
                ca = self.eval_network(val_data, val_class)
                validation_CAs.append((epoch, ca))

        return epoch_losses, validation_CAs


    def eval_network(self, data, data_class):
        """
        Evaluate the (so far) trained network on the provided data and classes/labels.

        Parameters:
            data - numpy array of dimensions [n0 x m], where m is the number of examples in 
                   the data, and n0 is the number of input attributes. The data is coming
                   either from the validation set, or test set.

            data_class - numpy array of dimensions [c x m], where c is the number of classes
        """
        
        n = data.shape[1]
        loss_avg = 0.0
        tp = 0.0

        for i in range(data.shape[1]):
            example = np.expand_dims(data[:, i], -1)  # one singular training example
            example_class = np.expand_dims(data_class[:, i], -1)
            example_class_num = np.argmax(data_class[:, i], axis=0)
            output, _, _ = self.forward_pass(example)  # TODO maybe just give data[:, i] here, see what fits nicer
            # output, _, _ = self.forward_pass(data[:, i])
            output_num = np.argmax(output, axis=0)[0]
            tp += int(example_class_num == output_num)

            loss = cross_entropy(example_class, output)
            loss_avg += loss

        print(f"Validation Loss: {loss_avg / n}")
        print(f"Classification accuracy: {tp / n}")
        return tp / n


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


if __name__ == "__main__":
    # Load CIFAR-10 dataset, make train, validation and test sets. 
    # CIFAR10:
    #   3072 input attributes, 10 classes
    #   50000 training samples, 10000 test samples, 6000 images per class
    train_file = "./data/train_data.pckl"
    test_file = "./data/test_data.pckl"
    train_data, train_class, test_data, test_class = load_data_cifar(train_file, test_file)
    val_pct = 0.1
    # val_size = int(len(train_data) * val_pct)  # error?
    val_size = int(train_data.shape[1] * val_pct)
    val_data = train_data[..., :val_size]
    val_class = train_class[..., :val_size]
    train_data = train_data[..., val_size:]
    train_class = train_class[..., val_size:]


    # Hyperparameters
    optimizer = "sgd"
    hidden_layers = [200, 200, 200]
    num_epochs = 20
    mini_batch_size = 64
    eta = 0.3
    lr_schedule = EXP_LR
    k = 0.001
    lmbd = 15  # regularization parameter or None, if we don't want regularization


    # Train and evaluate
    net = Network([train_data.shape[0],  # input layer
                   *hidden_layers,
                   train_class.shape[0]],  # output layer
                   optimizer=optimizer)
    epoch_losses, val_CAs = net.train(train_data, train_class, val_data, val_class,
                                      num_epochs, mini_batch_size,
                                      eta, lr_schedule=lr_schedule,
                                      k=k, lmbd=lmbd)
    print()
    print()
    print("=" * 50)
    print("=" * 50)
    net.eval_network(test_data, test_class)
    print()
    print("=" * 50)
    network_structure = f"\tOptimizer: \t\t{optimizer.upper()}\n" + \
                        f"\tHidden layers: \t\t{hidden_layers}\n" + \
                        f"\tNumber of epochs: \t{num_epochs}\n" + \
                        f"\tMini-batch size: \t{mini_batch_size}\n" + \
                        f"\tLearning rate: \t\t{eta}\n" + \
                        f"\tLR schedule: \t\t{lr_schedule}\n"
    if lr_schedule == EXP_LR:
        network_structure += f"\tDecay rate k: \t\t{k}\n"
    print(f"Network structure:\n{network_structure}")


    # Plot the average loss of each epoch, and classification accuracies on the validation set
    val_epochs, ca_values = list(zip(*val_CAs))
    val_epochs, ca_values = list(val_epochs), list(ca_values)
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 7))
    plt.plot(range(len(epoch_losses)), epoch_losses, label="Epoch losses")
    plt.plot(val_epochs, ca_values, label="CA on validation set")
    plt.xticks(range(0, len(epoch_losses)+1, 5))
    plt.legend()
    plt.show()
