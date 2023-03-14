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

    def __init__(self, sizes, optimizer="mbgd", eps=1e-8):
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
            self.m_weights = [np.zeros_like(w) for w in self.weights]
            self.v_weights = [np.zeros_like(w) for w in self.weights]
            self.m_biases = [np.zeros_like(b) for b in self.biases]
            self.v_biases = [np.zeros_like(b) for b in self.biases]
            self.eps = eps

    
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
            grad_weights - array of length num_layers-1 (same as self.weights), containing arrays of
                           shape [n_L+1, n_L] - same shape as weights for layer L in self.weights.
                           Each element of grad_weights contains the gradient of the loss function
                           w.r.t. the weights (i.e. the partial derivative of the loss function
                           w.r.t. all of the individual weights).

            grad_biases  - array of length num_layers-1 (same as self.biases), containing arrays of
                           shape [n_L+1, 1] - same shape as biases for layer L in self.biases.
                           Each element of grad_biases contains the gradient of the loss function
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
        grad_weights = [np.zeros(w.shape) for w in self.weights]
        grad_biases = [np.zeros(b.shape) for b in self.biases]


        # Backpropagation of the error through the layers
        num_weights = len(self.weights)
        for l in reversed(range(num_weights)):
            if l == num_weights - 1:  # final layer/output error
                delta = softmax_dLdZ(output, target)
            else:
                delta = np.dot(np.transpose(self.weights[l+1]), delta) * sigmoid_prime(Zs[l])
                
            grad_weights[l] = np.dot(delta, np.transpose(As[l])) / batch_size
            grad_biases[l] = np.sum(delta, axis=1, keepdims=True) / batch_size  # not axis=0!
        
        return grad_weights, grad_biases


    def update_network(self, grad_weights, grad_biases, eta, t, lmbd=0.0, n=None):
        """
        Update the weights and biases based on the used optimizer (MBGD or Adam).

        Parameters:
            grad_weights - array of length num_layers-1 (same as self.weights), containing arrays of
                           shape [n_L+1, n_L] - same shape as weights for layer L in self.weights.
                           Each element of grad_weights contains the gradient of the loss function
                           w.r.t. the weights (i.e. the partial derivative of the loss function
                           w.r.t. all of the individual weights). These gradients are calculated
                           during backward pass.

            grad_biases  - array of length num_layers-1 (same as self.biases), containing arrays of
                           shape [n_L+1, 1] - same shape as biases for layer L in self.biases.
                           Each element of grad_biases contains the gradient of the loss function
                           w.r.t. the biases (i.e. the partial derivative of the loss function
                           w.r.t. all of the individual biases). These gradients are calculated
                           during backward pass.

            eta         - (current) learning rate

            t           - current iteration, used for computing bias correction terms for Adam

            lmbd        - L2 regularization parameter

            n           - size of the training set, used when calculating the weight decays
        """

        weight_decays = [0] * len(self.weights)
        if lmbd != 0.0:
            if not n:
                raise ValueError("Size of training data (n) not given for L2 regularization when updating weights.")
            for i in range(len(self.weights)):
                weight_decays[i] = ((eta * lmbd) / n) * self.weights[i]

        if self.optimizer == "mbgd":
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] - weight_decays[i] - eta * grad_weights[i]
                self.biases[i] = self.biases[i] - eta * grad_biases[i]

        elif self.optimizer == "adam":
            for i in range(len(self.m_weights)):
                self.m_weights[i] = beta1 * self.m_weights[i] + (1 - beta1) * grad_weights[i]
                self.m_biases[i] = beta1 * self.m_biases[i] + (1 - beta1) * grad_biases[i]
                
                self.v_weights[i] = beta2 * self.v_weights[i] + (1 - beta2) * np.square(grad_weights[i])
                self.v_biases[i] = beta2 * self.v_biases[i] + (1 - beta2) * np.square(grad_biases[i])

                # Compute bias correction terms - makes learning slightly faster
                # https://stats.stackexchange.com/questions/232741/why-is-it-important-to-include-a-bias-correction-term-for-the-adam-optimizer-for
                # TODO ...

                adam_weights = self.m_weights[i] / (np.sqrt(self.v_weights[i]) + self.eps)
                adam_biases = self.m_biases[i] / (np.sqrt(self.v_biases[i]) + self.eps)

                self.weights[i] = self.weights[i] - weight_decays[i] - eta * adam_weights
                self.biases[i] = self.biases[i] - eta * adam_biases

        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")


    def train(self,
              training_data, training_class,
              val_data, val_class,
              epochs,
              mini_batch_size,
              eta, lr_schedule="no",
              k=0.001,
              lmbd=0.0):
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

            lr_schedule     - learning rate schedule, either "no" for no schedule, or "exponential"

            k               - the decay rate of learning rate, only used when lr_schedule == "exponential"

            lmbd            - L2 regularization parameter

        Returns:
            training_losses   - a list containing the (average) training loss for each epoch

            validation_losses - a list of tuples (epoch, validation_loss), containing validation losses 
                                whenever the network was tested against the validation set

            validation_cas    - a list of tuples (epoch, CA), containing classification accuracies whenever
                                the network was tested against the validation set
        """

        iteration_index = 0
        eta_current = eta
        n = training_data.shape[1]
        training_losses = []
        validation_cas = []
        validation_losses = []

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
                if (i+1) % 100 == 0: 
                    print(f"\tMini-batch {i+1} / {num}")

                output, Zs, As = self.forward_pass(mini_batch[0])
                grad_weights, grad_biases = \
                    self.backward_pass(output, mini_batch[1], Zs, As)

                self.update_network(grad_weights, grad_biases, eta_current, iteration_index, lmbd=lmbd, n=n)

                if lr_schedule == "no":
                    eta_current = eta
                elif lr_schedule == EXP_LR:
                    eta_current = eta * np.exp(-k * iteration_index)

                iteration_index += 1

                loss = cross_entropy(mini_batch[1], output, lmbd=lmbd, n=n, weights=self.weights)
                loss_avg += loss

            epoch_loss = loss_avg / len(mini_batches)
            training_losses.append(epoch_loss)
            print(f"Epoch {epoch + 1} complete")
            print(f"Loss: {epoch_loss}")

            if epoch == 0 or (epoch + 1) % 5 == 0:
                _, prev_validation_loss = validation_losses[-1] if len(validation_losses) else (-1, float("inf"))
                print(f"Previous validation loss: {prev_validation_loss}")
                valid_loss, valid_ca = self.eval_network(val_data, val_class)
                validation_losses.append((epoch, valid_loss))
                validation_cas.append((epoch, valid_ca))

        return training_losses, validation_losses, validation_cas


    def eval_network(self, data, data_class, test=False):
        """
        Evaluate the (so far) trained network on the provided data and classes/labels.

        Parameters:
            data       - numpy array of dimensions [n0 x m], where m is the number of examples in 
                         the data, and n0 is the number of input attributes. The data is coming
                         either from the validation set, or test set.

            data_class - numpy array of dimensions [c x m], where c is the number of classes
        """
        
        num_samples = data.shape[1]
        total_loss = 0.0
        tp = 0.0

        for i in range(data.shape[1]):
            example = np.expand_dims(data[:, i], -1)  # one singular training example
            example_class = np.expand_dims(data_class[:, i], -1)
            example_class_num = np.argmax(data_class[:, i], axis=0)
            output, _, _ = self.forward_pass(example)
            output_num = np.argmax(output, axis=0)[0]
            tp += int(example_class_num == output_num)

            loss = cross_entropy(example_class, output)
            total_loss += loss

        loss_avg = total_loss / num_samples
        CA = tp / num_samples
        if test:
            print(f"Current validation loss: {loss_avg}")
        else:
            print(f"Test loss: {loss_avg}")
        print(f"Classification accuracy: {CA}")
        return loss_avg, CA


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
    beta1           = 0.9  # beta1 for Adam optimizer
    beta2           = 0.999  # beta2 for Adam optimizer
    optimizer       = "mbgd"  # "mbgd" or "adam"
    hidden_layers   = [500, 500]  # structure of hidden layers
    num_epochs      = 50  # number of epochs
    mini_batch_size = 128  # size of mini_batch
    eta             = 0.3  # learning rate; when optimizer == "adam", set to 0.001 by default
    # lr_schedule     = EXP_LR
    lr_schedule     = "no"
    k               = 0.001  # decay rate when using exponential LR decay
    lmbd            = 0.1  # regularization parameter or 0.0, if we don't want regularization
    # λ smaller => we prefer to minimize the original cost function, λ bigger => we prefer to minimize the weights

    network_settings = f"\tOptimizer: \t\t{optimizer.upper()}\n" + \
                       f"\tHidden layers: \t\t{hidden_layers}\n" + \
                       f"\tNumber of epochs: \t{num_epochs}\n" + \
                       f"\tMini-batch size: \t{mini_batch_size}\n" + \
                       f"\tLearning rate: \t\t{eta}\n" + \
                       f"\tLR schedule: \t\t{lr_schedule}\n"
    if lr_schedule == EXP_LR:
        network_settings += f"\tDecay rate k: \t\t{k}\n"
    if lmbd != 0.0:
        network_settings += f"\tLambda (for L2): \t{lmbd}\n"
    if optimizer == "adam":
        network_settings += f"\tBetas (Adam): \t\tbeta1={beta1}, beta2={beta2}\n"


    # Train
    net = Network([train_data.shape[0],  # input layer
                   *hidden_layers,
                   train_class.shape[0]],  # output layer
                   optimizer=optimizer)
    print(f"Network settings:\n{network_settings}")

    train_losses, valid_losses, valid_cas = \
        net.train(train_data, train_class, val_data, val_class,
                  num_epochs, mini_batch_size,
                  eta, lr_schedule=lr_schedule,
                  k=k, lmbd=lmbd)
    

    # Evaluate
    print()
    print()
    print("=" * 50)
    print("=" * 50)
    net.eval_network(test_data, test_class, test=True)
    print()
    print(f"Network settings:\n{network_settings}")


    # Plot training losses, and classification accuracies and losses on the validation set
    valid_epochs, ca_values = list(zip(*valid_cas))
    valid_epochs, ca_values = list(valid_epochs), list(ca_values)
    _, loss_values = list(zip(*valid_losses))
    loss_values = list(loss_values)
    plt.style.use("ggplot")

    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(10, 7)
    ax1.plot(range(len(train_losses)), train_losses)
    ax1.set_xticks(range(0, len(train_losses)+1, 5))
    fig1.suptitle("Training losses")

    fig2, ax2 = plt.subplots()
    fig2.set_size_inches(10, 7)
    ax2.plot(valid_epochs, ca_values, color="red", label="Classification accuracies")
    ax2.plot(valid_epochs, loss_values, color="blue", label="Losses")
    ax2.set_xticks(range(0, len(train_losses)+1, 5))
    fig2.suptitle("CAs and losses on validation set")

    plt.show()
