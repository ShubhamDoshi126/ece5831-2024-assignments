import numpy as np
import pickle

class Mnist:
    """
    A class to represent the MNIST model for digit recognition.

    Attributes:
        params (dict): Dictionary containing the model parameters (weights).

    Methods:
        sigmoid(x):
            Apply the sigmoid activation function.
        predict(image):
            Predict the digit label for a given 28x28 grayscale image.
    """

    def __init__(self):
        """
        Initialize the Mnist class by loading pre-trained weights from 'sample_weight.pkl'.
        """
        # Load pre-trained weights
        with open('sample_weight.pkl', 'rb') as f:
            self.params = pickle.load(f)
            print("Weights loaded successfully")
    
    def sigmoid(self, x):
        """
        Apply the sigmoid activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array after applying the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))
    
    def predict(self, image):
        """
        Predict the digit label for a given 28x28 grayscale image.

        Args:
            image (np.ndarray): Input image of shape (28, 28).

        Returns:
            int: Predicted digit label.
        
        Raises:
            ValueError: If the input image is not of shape (28, 28).
        """
        # Ensure image is in correct format (28x28)
        if image.shape != (28, 28):
            raise ValueError(f"Image must be 28x28 grayscale, got shape {image.shape}")
        
        # Preprocess image and flatten
        image = image.flatten()
        
        # Forward pass through the network
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
        
        a1 = np.dot(image, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = self.sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = self.sigmoid(a3)
        
        # Return the index of the highest probability
        return np.argmax(y)