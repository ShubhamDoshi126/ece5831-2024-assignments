import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

class Reuters:
    def __init__(self):
        self.num_words = 10000
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.model = None
        self.history = None

    def prepare_data(self):
        """Prepare Reuters data for training and evaluation."""
        # Load Reuters dataset
        (self.train_data, self.train_labels), (self.test_data, 
        self.test_labels) = keras.datasets.reuters.load_data(
            num_words=self.num_words
        )
        
        # Vectorize data
        self.x_train = self.vectorize_sequences(self.train_data)
        self.x_test = self.vectorize_sequences(self.test_data)

        # One-hot encode labels
        self.y_train = self.to_one_hot(self.train_labels)
        self.y_test = self.to_one_hot(self.test_labels)

        # Create validation set
        self.x_val = self.x_train[:1000]
        self.partial_x_train = self.x_train[1000:]

        self.y_val = self.y_train[:1000]
        self.partial_y_train = self.y_train[1000:]

    def vectorize_sequences(self, sequences, dimension=10000):
        """Create all-zero matrix and set matching indices to 1."""
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

    def to_one_hot(self, labels, dimension=46):
        """Convert labels to one-hot encoded vectors."""
        results = np.zeros((len(labels), dimension))
        for i, label in enumerate(labels):
            results[i, label] = 1.
        return results

    def build_model(self):
        """Build a neural network model for multiclass classification."""
        # Define the model architecture
        self.model = keras.Sequential(
            [
                layers.Dense(64, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(46, activation="softmax")
            ]
        )
        
        # Compile the model with optimizer, loss function, and metrics
        self.model.compile(optimizer="rmsprop",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        
    def train(self, epochs=20, batch_size=512):
        """Train the model using training and validation data."""
        self.history = self.model.fit(self.partial_x_train,
                    self.partial_y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(self.x_val, self.y_val))

    def plot_loss(self):
        """Plot the training and validation loss."""
        history_dict = self.history.history
        loss_values = history_dict["loss"]
        val_loss_values = history_dict["val_loss"]
        epochs = range(1, len(loss_values) + 1)

        plt.plot(epochs, loss_values, "bo", label="Training loss")
        plt.plot(epochs, val_loss_values, "b", label="Validation loss")
        plt.title("Training and validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_accuracy(self):
        """Plot the training and validation accuracy."""
        history_dict = self.history.history
        acc = history_dict["accuracy"]
        val_acc = history_dict["val_accuracy"]
        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, "bo", label="Training acc")
        plt.plot(epochs, val_acc, "b", label="Validation acc")
        plt.title("Training and validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    def evaluate(self):
        """Evaluate the model on the test data."""
        results = self.model.evaluate(self.x_test, self.y_test)
        print("Loss and accuracy on test data:", results)