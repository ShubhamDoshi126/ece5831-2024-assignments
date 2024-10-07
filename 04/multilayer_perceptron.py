import numpy as np

class MultiLayerPerceptron:
    """
    A simple implementation of a Multi-Layer Perceptron neural network.
    
    This implementation includes:
    - Three layers with hardcoded weights and biases
    - Sigmoid activation for hidden layers
    - Identity activation for output layer
    """
    def __init__(self):
        self.net = {}
        pass

    def init_network(self):
        net = {}
        # for layer 1
        net['w1'] = np.array([[0.7, 0.9, 0.3], [0.5, 0.4, 0.1]])
        net['b1'] = np.array([1, 1, 1])
        # for layer 2
        net['w2'] = np.array([[0.2, 0.3], [0.4, 0.5], [0.22, 0.1234]])
        net['b2'] = np.array([0.5, 0.5])
        # for layer 3 (output)
        net['w3'] = np.array([[0.7, 0.1], [0.123, 0.314]])
        net['b3'] = np.array([0.1, 0.2])

        self.net = net

    def forward(self, x):
        w1, w2, w3 = self.net['w1'], self.net['w2'], self.net['w3']
        b1, b2, b3 = self.net['b1'], self.net['b2'], self.net['b3']

        # Layer 1
        a1 = np.dot(x, w1) + b1
        z1 = self.sigmoid(a1)
        print(f"Layer 1 activation (a1): {a1}")
        print(f"Layer 1 output (z1): {z1}")

        # Layer 2
        a2 = np.dot(z1, w2) + b2
        z2 = self.sigmoid(a2)
        print(f"Layer 2 activation (a2): {a2}")
        print(f"Layer 2 output (z2): {z2}")

        # Layer 3 (output)
        a3 = np.dot(z2, w3) + b3
        y = self.identity(a3)
        print(f"Layer 3 activation (a3): {a3}")
        print(f"Final output (y): {y}")

        return y

    def identity(self, x):
        return x

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


