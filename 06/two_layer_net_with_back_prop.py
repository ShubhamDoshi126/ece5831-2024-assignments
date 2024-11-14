import numpy as np
from activations import Activations
from errors import Errors
from layers import Affine, Relu, SoftmaxWithLoss
from collections import OrderedDict

class TwoLayerNetWithBackProp:
    """
    A two-layer neural network with backpropagation.

    This network consists of an input layer, a hidden layer with ReLU activation,
    and an output layer using softmax with loss for multi-class classification.
    """

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        Initialize the network with random weights and biases and create the layer structure.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of output classes.
            weight_init_std (float): Standard deviation for weight initialization.
        """
        # Initialize parameters
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)  # Weight for layer 1
        self.params['b1'] = np.zeros(hidden_size)  # Bias for layer 1
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)  # Weight for layer 2
        self.params['b2'] = np.zeros(output_size)  # Bias for layer 2

        # Initialize auxiliary components and ordered layers
        self.activations = Activations()
        self.errors = Errors()

        # Define the sequence of layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['w1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['w2'], self.params['b2'])
        self.last_layer = SoftmaxWithLoss()

        print("TwoLayerNetWithBackProp initialized successfully!")

    def predict(self, x):
        """
        Perform forward propagation to predict the output.

        Args:
            x (np.ndarray): Input data.
        
        Returns:
            np.ndarray: Network output after forward propagation.
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, y):
        """
        Calculate the loss of the network for given input and true labels.

        Args:
            x (np.ndarray): Input data.
            y (np.ndarray): True labels.
        
        Returns:
            float: Loss value calculated by softmax and cross-entropy.
        """
        y_hat = self.predict(x)  # Forward pass to get predictions
        return self.last_layer.forward(y_hat, y)

    def accuracy(self, x, y):
        """
        Calculate the accuracy of the network for given input and true labels.

        Args:
            x (np.ndarray): Input data.
            y (np.ndarray): True labels in one-hot encoded format.
        
        Returns:
            float: Accuracy score as a fraction between 0 and 1.
        """
        y_hat = self.predict(x)  # Forward pass to get predictions
        p = np.argmax(y_hat, axis=1)  # Predicted labels
        y_p = np.argmax(y, axis=1)    # True labels
        return np.sum(p == y_p) / float(x.shape[0])  # Accuracy calculation
    
    def gradient(self, x, y):
        """
        Compute gradients of weights and biases with respect to the loss function using backpropagation.

        Args:
            x (np.ndarray): Input data.
            y (np.ndarray): True labels.
        
        Returns:
            dict: Gradients for each parameter.
        """
        # Forward and backward passes for backpropagation
        self.loss(x, y)
        dout = 1
        dout = self.last_layer.backward(dout)

        # Backward pass through layers in reverse order
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # Store gradients for parameters
        grads = {}
        grads['w1'] = self.layers['Affine1'].dw
        grads['b1'] = self.layers['Affine1'].db
        grads['w2'] = self.layers['Affine2'].dw
        grads['b2'] = self.layers['Affine2'].db

        return grads

if __name__ == "__main__":
    # Create an instance of the TwoLayerNetWithBackProp class for testing
    model = TwoLayerNetWithBackProp(input_size=784, hidden_size=100, output_size=10)
    
    # Print a success message if initialization is successful
    print("Class ran successfully.")
