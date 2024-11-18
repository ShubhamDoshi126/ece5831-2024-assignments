import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
import numpy as np

class LeNet:
    """
    Implementation of the LeNet Convolutional Neural Network for digit classification on the MNIST dataset.
    
    Attributes:
        batch_size (int): Number of samples per gradient update.
        epochs (int): Number of training iterations over the entire dataset.
        model (Sequential): The LeNet architecture implemented using TensorFlow's Sequential API.
    """
    def __init__(self, batch_size=64, epochs=5):
        """
        Initialize the LeNet class with batch size, number of epochs, and constructs the LeNet model.
        
        Args:
            batch_size (int): The size of the training batch. Default is 64.
            epochs (int): The number of epochs to train. Default is 5.
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self._create_lenet()
        self._compile()

    def _create_lenet(self):
        """
        Define the architecture of the LeNet model using the Sequential API.
        """
        self.model = Sequential([
            # First convolutional layer with 6 filters, 5x5 kernel, sigmoid activation, and 'same' padding
            Conv2D(6, (5, 5), activation='sigmoid', input_shape=(28, 28, 1), padding='same'),
            # First average pooling layer with 2x2 pool size and stride of 2
            AveragePooling2D(pool_size=(2, 2), strides=2),
            # Second convolutional layer with 16 filters, 5x5 kernel, sigmoid activation, and 'same' padding
            Conv2D(16, (5, 5), activation='sigmoid', padding='same'),
            # Second average pooling layer with 2x2 pool size and stride of 2
            AveragePooling2D(pool_size=(2, 2), strides=2),
            # Flatten layer to convert 2D feature maps into 1D feature vectors
            Flatten(),
            # Fully connected layer with 120 neurons and sigmoid activation
            Dense(120, activation='sigmoid'),
            # Fully connected layer with 84 neurons and sigmoid activation
            Dense(84, activation='sigmoid'),
            # Output layer with 10 neurons (for 10 classes) and softmax activation
            Dense(10, activation='softmax')
        ])

    def _compile(self):
        """
        Compile the LeNet model with an optimizer, loss function, and evaluation metrics.
        """
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def _preprocess(self):
        """
        Load and preprocess the MNIST dataset for training and testing.
        
        - Normalize pixel values to the range [0, 1].
        - Reshape images to include the channel dimension.
        - Convert labels to one-hot encoded format.
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # Normalize the pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        # Reshape images to include a single channel (grayscale)
        self.x_train = x_train.reshape(-1, 28, 28, 1)
        self.x_test = x_test.reshape(-1, 28, 28, 1)
        # One-hot encode the labels
        self.y_train = to_categorical(y_train, 10)
        self.y_test = to_categorical(y_test, 10)

    def train(self):
        """
        Train the LeNet model using the preprocessed training data.
        """
        self._preprocess()  # Load and preprocess the data
        self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs)

    def save(self, model_path_name):
        """
        Save the trained model to a specified file path.
        
        Args:
            model_path_name (str): The file path to save the model.
        """
        self.model.save(f"{model_path_name}.keras")

    def load(self, model_path_name):
        """
        Load a previously saved model from the specified file path.
        
        Args:
            model_path_name (str): The file path from where to load the model.
        """
        self.model = tf.keras.models.load_model(f"{model_path_name}.keras")

    def predict(self, images):
        """
        Predict the class probabilities for a given set of images.
        
        Args:
            images (numpy.ndarray): A batch of input images for prediction.
        
        Returns:
            numpy.ndarray: The predicted class probabilities for each image.
        """
        return self.model.predict(images)

if __name__ == "__main__":
    # Initialize LeNet with default parameters
    lenet = LeNet(batch_size=64, epochs=5)

    # Train the model on the MNIST dataset
    print("Training the LeNet model...")
    lenet.train()

    # Save the trained model to a file
    model_file = "doshi_cnn_model"
    print(f"Saving the model to {model_file}.keras...")
    lenet.save(model_file)

    # Load the saved model for verification
    print("Loading the saved model...")
    lenet.load(model_file)

    print("Model trained, saved, and reloaded successfully!")
