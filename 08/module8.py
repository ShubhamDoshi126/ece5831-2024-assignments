import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from le_net import LeNet


def load_image(filename):
    """
    Loads and preprocesses an image file for prediction.

    Args:
        filename (str): The path to the image file.

    Returns:
        numpy.ndarray: A reshaped numpy array of the image in grayscale,
                       normalized to a [0, 1] range.
    """
    # Open the image and convert it to grayscale
    image = Image.open(filename).convert('L')
    # Resize the image to match the input size expected by the model (28x28 pixels)
    image = image.resize((28, 28))
    # Convert the image to a numpy array and normalize pixel values
    image_np = np.array(image).astype(np.float32) / 255.0
    # Reshape the image to add the channel dimension (1 channel for grayscale)
    return image_np.reshape(1, 28, 28, 1)


def main():
    """
    Main function to load an image, load the trained model, and perform digit recognition.

    Command-line Arguments:
        1. image_filename (str): The name of the image file to be processed.
        2. expected_digit (int): The expected digit (0-9) that is represented in the image.

    Outputs:
        - Displays the input image with prediction results.
        - Prints success or failure message based on prediction accuracy.
    """
    # Ensure the correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python module8.py <image_filename> <expected_digit>")
        sys.exit(1)

    # Parse command-line arguments
    image_filename = sys.argv[1]
    expected_digit = int(sys.argv[2])

    # Construct the full image file path based on the expected digit directory structure
    base_dir = os.path.join(os.path.dirname(__file__), "Custom MNIST Samples")
    full_image_path = os.path.join(base_dir, f"Digit {expected_digit}", image_filename)

    # Check if the image file exists
    if not os.path.exists(full_image_path):
        print(f"Error: Image file {full_image_path} not found.")
        sys.exit(1)

    # Load the trained LeNet model
    model_path = 'doshi_cnn_model'
    lenet = LeNet()
    try:
        lenet.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found.")
        sys.exit(1)

    # Load and preprocess the image for prediction
    image = load_image(full_image_path)
    if image is None:
        print("Error: Failed to load or preprocess the image.")
        sys.exit(1)

    # Predict the digit using the trained model
    predictions = lenet.predict([image])
    predicted_digit = np.argmax(predictions[0])  # Identify the class with the highest probability

    # Display the image with prediction results
    plt.imshow(image.reshape(28, 28), cmap='gray')  # Reshape the image for visualization
    plt.title(f"Predicted: {predicted_digit}, Expected: {expected_digit}")
    plt.axis('off')  # Turn off axis labels for better visualization
    plt.show()

    # Print the result of the prediction (success or failure)
    if predicted_digit == expected_digit:
        print(f"Success: Image {image_filename} is recognized as {predicted_digit}.")
    else:
        print(f"Fail: Image {image_filename} is for digit {expected_digit} but the inference result is {predicted_digit}.")


if __name__ == "__main__":
    main()
