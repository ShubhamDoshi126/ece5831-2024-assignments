import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

class IMDb:
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
        """Prepare IMDB data for training and evaluation."""
        # Load IMDB dataset [1]
        (self.train_data, self.train_labels), (self.test_data, 
        self.test_labels) = keras.datasets.imdb.load_data(
            num_words=self.num_words
        )
        
        # Vectorize data [2, 3]
        self.x_train = self.vectorize_sequences(self.train_data)
        self.x_test = self.vectorize_sequences(self.test_data)

        # Vectorize labels [3]
        self.y_train = np.asarray(self.train_labels).astype("float32")
        self.y_test = np.asarray(self.test_labels).astype("float32")

        # Create validation set [4]
        self.x_val = self.x_train[:10000]
        self.partial_x_train = self.x_train[10000:]

        self.y_val = self.y_train[:10000]
        self.partial_y_train = self.y_train[10000:]

    def vectorize_sequences(self, sequences, dimension=10000):
        """Create all-zero matrix and set matching indices to 1."""
        results = np.zeros((len(sequences), dimension)) # [2]
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.  # [2]
        return results

    def build_model(self):
        """Build a neural network model for binary classification."""
        # Define the model architecture [3]
        self.model = keras.Sequential(
            [
                layers.Dense(16, activation="relu"), # [3]
                layers.Dense(16, activation="relu"), # [3]
                layers.Dense(1, activation="sigmoid") # [3]
            ]
        )
        
        # Compile the model with optimizer, loss function, and metrics [4, 5]
        self.model.compile(optimizer="rmsprop", # [5]
                      loss="binary_crossentropy", # [5]
                      metrics=["accuracy"]) # [4]
        
    def train(self, epochs=20, batch_size=512):
        """Train the model using training and validation data."""
        self.history = self.model.fit(self.partial_x_train,
                    self.partial_y_train,
                    epochs=epochs, # [4]
                    batch_size=batch_size, # [4]
                    validation_data=(self.x_val, self.y_val)) # [4]

    def plot_loss(self):
        """Plot the training and validation loss."""
        history_dict = self.history.history # [6]
        loss_values = history_dict["loss"] # [6]
        val_loss_values = history_dict["val_loss"] # [6]
        epochs = range(1, len(loss_values) + 1) # [6]

        plt.plot(epochs, loss_values, "bo", label="Training loss") # [7]
        plt.plot(epochs, val_loss_values, "b", label="Validation loss") # [7]
        plt.title("Training and validation loss") # [7]
        plt.xlabel("Epochs") # [7]
        plt.ylabel("Loss") # [7]
        plt.legend() # [7]
        plt.show() # [7]

    def plot_accuracy(self):
        """Plot the training and validation accuracy."""
        history_dict = self.history.history
        acc = history_dict["accuracy"] # [8]
        val_acc = history_dict["val_accuracy"] # [8]
        epochs = range(1, len(acc) + 1) # [8]

        plt.plot(epochs, acc, "bo", label="Training acc") # [8]
        plt.plot(epochs, val_acc, "b", label="Validation acc") # [8]
        plt.title("Training and validation accuracy") # [8]
        plt.xlabel("Epochs") # [8]
        plt.ylabel("Accuracy") # [8]
        plt.legend() # [8]
        plt.show() # [8]

    def evaluate(self):
        """Evaluate the model on the test data."""
        results = self.model.evaluate(self.x_test, self.y_test) # [7]
        print("Loss and accuracy on test data:", results) # [7]