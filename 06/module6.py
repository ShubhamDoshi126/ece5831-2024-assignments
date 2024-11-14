import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle

def load_image(filename):
    """
    Loads and preprocesses an image file for prediction.

    Args:
        filename (str): The path to the image file.

    Returns:
        numpy.ndarray: A reshaped numpy array of the image in grayscale,
                       normalized to a [0, 1] range and flattened.
    """
    # Open the image and convert it to grayscale
    image = Image.open(filename).convert('L')
    # Resize the image to match the input size expected by the model (28x28 pixels)
    image = image.resize((28, 28))
    # Convert the image to a numpy array and normalize pixel values
    image_np = np.array(image).astype(np.float32) / 255.0
    # Flatten the image to a 1D array for input to the model
    return image_np.reshape(1, -1)

def main():
    """
    Main function to load an image, load the trained model, and perform digit recognition.

    The function takes two command-line arguments: the filename of the image and the expected digit.
    It displays the image, predicts the digit using the trained model, and prints whether
    the prediction matches the expected digit.

    Command-line Arguments:
        image_filename (str): The name of the image file to be processed (e.g., "0_0.png").
        expected_digit (int): The expected digit (0-9) that is represented in the image.

    Raises:
        IndexError: If insufficient command-line arguments are provided.
        FileNotFoundError: If the specified model file or image file cannot be found.
    """
    try:
        # Get the image filename and expected digit from command-line arguments
        image_filename = sys.argv[1]
        expected_digit = int(sys.argv[2])
    except IndexError:
        print("Usage: python module6.py <image_filename> <expected_digit>")
        sys.exit(1)

    # Define the path to the image based on expected digit directory structure
    base_dir = os.path.join(os.path.dirname(__file__), "Custom MNIST Samples")
    full_image_path = os.path.join(base_dir, f"Digit {expected_digit}", image_filename)

    # Load the trained model from a pickle file
    model_path = 'doshi_mnist_model.pkl'
    try:
        with open(model_path, 'rb') as f:
            mnist_model = pickle.load(f)
    except FileNotFoundError:
        print(f"Model file {model_path} not found.")
        sys.exit(1)

    # Load the image to be tested
    image = load_image(full_image_path)
    if image is None:
        print("Failed to load the image.")
        return

    # Predict the digit in the image using the loaded model
    prediction = mnist_model.predict(image)
    predicted_digit = np.argmax(prediction)  # Extract the predicted class

    # Display the image with prediction results
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_digit}, Expected: {expected_digit}")
    plt.axis('off')
    plt.show()

    # Print a success or failure message based on prediction accuracy
    if predicted_digit == expected_digit:
        print(f"Success: Image {image_filename} is recognized as {predicted_digit}.")
    else:
        print(f"Fail: Image {image_filename} is for digit {expected_digit} but the inference result is {predicted_digit}.")

if __name__ == "__main__":
    main()
