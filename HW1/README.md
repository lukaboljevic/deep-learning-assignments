# Network implementation

Implementation of a simple multi layer feed forward neural network using no external libraries (other than Numpy). The following things are included:
- He weight initialization
- SGD (I called it MBGD or mini-batch gradient descent, but as far as I can see, people use those two terms interchangeably) and Adam optimizer
- Sigmoid + softmax (on the last layer only) activation functions
- Cross entropy loss
- L2 regularization
- Exponential learning rate decay

Data: https://drive.google.com/file/d/17EXf64CJpjSAEhb5y6AX42Z0iYn-IvVV/view
