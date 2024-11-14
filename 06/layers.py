import numpy as np
from activations import Activations
from errors import Errors

class Relu:
    """
    ReLU (Rectified Linear Unit) activation layer for a neural network.
    Applies the ReLU function element-wise.
    """
    def __init__(self):
        self.mask = None  # Mask for identifying negative inputs

    def forward(self, x):
        """
        Forward pass for ReLU.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output with ReLU applied (negative values set to 0).
        """
        # Create mask for negative values
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0  # Set negative values to 0
        return out

    def backward(self, dout):
        """
        Backward pass for ReLU.

        Args:
            dout (numpy.ndarray): Upstream gradient.

        Returns:
            numpy.ndarray: Gradient of ReLU with respect to input.
        """
        # Apply mask to the gradient
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    """
    Sigmoid activation layer for a neural network.
    """
    def __init__(self):
        self.out = None  # Store the output of sigmoid for backpropagation
        self.activations = Activations()

    def forward(self, x):
        """
        Forward pass for Sigmoid.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output after applying sigmoid activation.
        """
        # Apply sigmoid and store output
        out = self.activations.sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        """
        Backward pass for Sigmoid.

        Args:
            dout (numpy.ndarray): Upstream gradient.

        Returns:
            numpy.ndarray: Gradient of Sigmoid with respect to input.
        """
        # Compute derivative of sigmoid
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine:
    """
    Affine (Fully Connected) layer for a neural network.
    """
    def __init__(self, w, b):
        """
        Initializes weights and biases for the affine layer.

        Args:
            w (numpy.ndarray): Weights of shape (input_dim, output_dim).
            b (numpy.ndarray): Biases of shape (output_dim,).
        """
        self.w = w  # Weights
        self.b = b  # Biases
        self.x = None  # Input data
        self.original_x_shape = None  # Original shape of input data
        self.dw = None  # Gradient of weights
        self.db = None  # Gradient of biases

    def forward(self, x):
        """
        Forward pass for Affine layer.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output of the affine layer.
        """
        # Save the original shape of x for backpropagation
        self.original_x_shape = x.shape
        # Flatten input if necessary
        x = x.reshape(x.shape[0], -1)
        self.x = x
        # Compute affine transformation
        out = np.dot(self.x, self.w) + self.b
        return out

    def backward(self, dout):
        """
        Backward pass for Affine layer.

        Args:
            dout (numpy.ndarray): Upstream gradient.

        Returns:
            numpy.ndarray: Gradient with respect to input.
        """
        # Compute gradients for weights and biases
        dx = np.dot(dout, self.w.T)  # Gradient with respect to input
        self.dw = np.dot(self.x.T, dout)  # Gradient with respect to weights
        self.db = np.sum(dout, axis=0)  # Gradient with respect to biases
        # Reshape dx to match the original input shape
        dx = dx.reshape(*self.original_x_shape)
        return dx


class SoftmaxWithLoss:
    """
    Softmax activation combined with Cross-Entropy Loss for neural network output layer.
    """
    def __init__(self):
        self.loss = None     # Cross-entropy loss
        self.y_hat = None    # Predicted probabilities (softmax output)
        self.y = None        # Ground truth labels
        self.activations = Activations()
        self.errors = Errors()

    def forward(self, x, y):
        """
        Forward pass for Softmax with Cross-Entropy Loss.

        Args:
            x (numpy.ndarray): Input data (logits).
            y (numpy.ndarray): True labels.

        Returns:
            float: Loss value.
        """
        # Store true labels and apply softmax to logits
        self.y = y
        self.y_hat = self.activations.softmax(x)
        # Compute cross-entropy loss
        self.loss = self.errors.cross_entropy_error(self.y_hat, self.y)
        return self.loss

    def backward(self, dout=1):
        """
        Backward pass for Softmax with Cross-Entropy Loss.

        Args:
            dout (float): Upstream gradient (default is 1).

        Returns:
            numpy.ndarray: Gradient with respect to input.
        """
        # Calculate gradient of loss with respect to input
        batch_size = self.y.shape[0]
        dx = (self.y_hat - self.y) / batch_size
        return dx
