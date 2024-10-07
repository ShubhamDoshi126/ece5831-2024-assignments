import numpy as np
from multilayer_perceptron import MultiLayerPerceptron

# Initialize and test the network 
mlp = MultiLayerPerceptron()
mlp.init_network()
y = mlp.forward(np.array([7.0, 2.0]))  # Test with some input parameters
print(f"Network output: {y}")
