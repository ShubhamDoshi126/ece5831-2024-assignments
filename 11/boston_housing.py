import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

class BostonHousing:
    def __init__(self):
        self.train_data = None
        self.train_targets = None
        self.test_data = None
        self.test_targets = None
        self.model = None
        self.history = None
        self.k = 4
        self.num_val_samples = None
        self.num_epochs = 500
        self.all_mae_histories = []
        

    def prepare_data(self):
        """Prepare data for training and evaluation."""
        (self.train_data, self.train_targets), (self.test_data, 
        self.test_targets) = keras.datasets.boston_housing.load_data()

        # Normalize the data
        mean = self.train_data.mean(axis=0)
        self.train_data -= mean
        std = self.train_data.std(axis=0)
        self.train_data /= std

        self.test_data -= mean
        self.test_data /= std

        self.num_val_samples = len(self.train_data) // self.k

    def build_model(self):
        """Build the neural network model."""
        self.model = keras.Sequential(
            [
                layers.Dense(64, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(1)  # No activation for regression
            ]
        )
        self.model.compile(optimizer="rmsprop",
                      loss="mse",  # Mean squared error for regression
                      metrics=["mae"])  # Mean absolute error for evaluation

    def train(self):
        """Train the model using k-fold validation."""
        for i in range(self.k):
            print(f"Processing fold #{i}")
            val_data = self.train_data[i * self.num_val_samples: (i + 1) 
            * self.num_val_samples]
            val_targets = self.train_targets[i * self.num_val_samples: (i + 1) 
            * self.num_val_samples]

            partial_train_data = np.concatenate(
                [self.train_data[:i * self.num_val_samples],
                 self.train_data[(i + 1) * self.num_val_samples:]],
                axis=0
            )
            partial_train_targets = np.concatenate(
                [self.train_targets[:i * self.num_val_samples],
                 self.train_targets[(i + 1) * self.num_val_samples:]],
                axis=0
            )

            self.build_model()  # Create a fresh model for each fold
            history = self.model.fit(
                partial_train_data,
                partial_train_targets,
                epochs=self.num_epochs,
                batch_size=16,
                verbose=0,
                validation_data=(val_data, val_targets),
            )
            mae_history = history.history["val_mae"]
            self.all_mae_histories.append(mae_history)

    def plot_loss(self):
        """Plot validation MAE across epochs, averaging over folds."""
        average_mae_history = [
            np.mean([x[i] for x in self.all_mae_histories]) 
            for i in range(self.num_epochs)
        ]
        plt.plot(range(1, len(average_mae_history) + 1), 
        average_mae_history)
        plt.xlabel("Epochs")
        plt.ylabel("Validation MAE")
        plt.show()

    def evaluate(self):
        """Evaluate the model on the test data."""
        test_mse_score, test_mae_score = self.model.evaluate(
            self.test_data, self.test_targets
        )
        print(f"Test MAE: {test_mae_score:.2f}")