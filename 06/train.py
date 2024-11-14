import numpy as np
import pickle
from two_layer_net_with_back_prop import TwoLayerNetWithBackProp
from mnist import Mnist

# Hyperparameters
iterations = 10000
batch_size = 16
learning_rate = 0.01
output_file = 'doshi_mnist_model.pkl'

# Load MNIST data
mnist = Mnist()
(x_train, y_train), (x_test, y_test) = mnist.load()
train_size = x_train.shape[0]
iter_per_epoch = max(train_size // batch_size, 1)

# Initialize the two-layer network
network = TwoLayerNetWithBackProp(input_size=784, hidden_size=100, output_size=10)

# Lists to record training progress
train_loss_list = []
train_acc_list = []
test_acc_list = []

# Training loop
for i in range(iterations):
    """
    Train the two-layer neural network with mini-batch gradient descent.
    
    Steps:
        1. Select a mini-batch from the training data.
        2. Compute gradients for backpropagation.
        3. Update network parameters based on computed gradients.
        4. Record training loss and accuracy at each epoch.
    
    Args:
        iterations (int): Total number of iterations for training.
    """
    # Mini-batch selection
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]

    # Compute gradients
    grad = network.gradient(x_batch, y_batch)

    # Update parameters
    for key in network.params:
        network.params[key] -= learning_rate * grad[key]

    # Record training progress
    loss = network.loss(x_batch, y_batch)
    train_loss_list.append(loss)

    # Record accuracy at each epoch
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, y_train)
        test_acc = network.accuracy(x_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"Iteration {i}, Training accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")

# Save the trained model
with open(output_file, 'wb') as f:
    """
    Save the trained model to a file using pickle.
    
    Args:
        output_file (str): File path where the model will be saved.
    """
    pickle.dump(network, f)
print(f"Model saved as {output_file}")
