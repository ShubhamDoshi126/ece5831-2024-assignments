import sys
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Constants for model paths
MODEL_PATH = r"C:\Users\shubdosh\Desktop\ece5831\05\model\keras_model.h5"
LABELS_PATH = r"C:\Users\shubdosh\Desktop\ece5831\05\model\labels.txt"

def load_labels(label_file):
    """
    Loads class labels from a text file.

    Args:
        label_file (str): Path to the file containing class labels, with each label on a new line.
    
    Returns:
        list: A list of class names.
    """
    with open(label_file, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def main(image_path):
    """
    Main function to load an image, preprocess it, make a prediction using the pre-trained model, 
    and display the result.

    Args:
        image_path (str): Path to the image file that needs to be classified.

    Returns:
        None
    """
    # Verify image path
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        sys.exit(1)

    # Load model and labels
    model = tf.keras.models.load_model(MODEL_PATH)
    """
    Loads a pre-trained Keras model from the specified file.

    Returns:
        model (tensorflow.python.keras.engine.functional.Functional): The loaded Keras model.
    """
    class_names = load_labels(LABELS_PATH)
    """
    Loads the class labels from the specified text file.

    Returns:
        class_names (list): A list of class names corresponding to the model's output classes.
    """

    # Load and preprocess image
    img = cv2.imread(image_path)
    """
    Loads the image from the provided path using OpenCV.

    Returns:
        img (numpy.ndarray): The loaded image in BGR format.
    """
    if img is None:
        print(f"Error: Failed to load image from '{image_path}'.")
        sys.exit(1)

    img_resized = cv2.resize(img, (224, 224))  # Assuming the model uses 224x224 input
    """
    Resizes the image to 224x224 pixels to match the input size expected by the model.

    Returns:
        img_resized (numpy.ndarray): The resized image.
    """
    img_normalized = img_resized.astype('float32') / 255.0
    """
    Normalizes the image pixel values to the range [0, 1].

    Returns:
        img_normalized (numpy.ndarray): The normalized image.
    """
    img_expanded = np.expand_dims(img_normalized, axis=0)
    """
    Adds a batch dimension to the image array so it can be fed to the model.

    Returns:
        img_expanded (numpy.ndarray): The preprocessed image ready for model input.
    """

    # Make prediction
    predictions = model.predict(img_expanded)
    """
    Runs the preprocessed image through the model to obtain predictions.

    Returns:
        predictions (numpy.ndarray): The prediction probabilities for each class.
    """
    class_idx = np.argmax(predictions)
    """
    Finds the index of the class with the highest prediction probability.

    Returns:
        class_idx (int): The index of the predicted class.
    """
    confidence_score = np.max(predictions)
    """
    Retrieves the confidence score for the predicted class.

    Returns:
        confidence_score (float): The confidence score of the predicted class.
    """

    print(f'Class: {class_names[class_idx]}')
    """
    Prints the predicted class name.
    """
    print(f'Confidence Score: {confidence_score:.4f}')
    """
    Prints the confidence score of the prediction.
    """

    # Display image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    """
    Converts the image from BGR to RGB and displays it using Matplotlib.
    """
    plt.title(f'Prediction: {class_names[class_idx]} ({confidence_score:.4f})')
    """
    Sets the title of the displayed image to the predicted class and confidence score.
    """
    plt.axis('off')
    """
    Hides the axes in the image display.
    """
    plt.show()
    """
    Shows the image with the predicted label.
    """

if __name__ == "__main__":
    """
    Entry point of the script. Checks the number of command-line arguments and calls the main function.
    """
    if len(sys.argv) != 2:
        print("Usage: python rock-paper-scissor.py <image_path>")
        sys.exit(1)
        
    main(sys.argv[1])
