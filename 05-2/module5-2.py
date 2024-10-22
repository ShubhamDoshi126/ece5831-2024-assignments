"""
MNIST Dataset Visualization Tool

This script provides functionality to visualize individual samples from the MNIST dataset.
It uses matplotlib for visualization and supports both training and test datasets.
"""

import argparse
import matplotlib.pyplot as plt
from mnist_data import MnistData

def visualize_sample(image, label, title):
    """
    Visualize a single MNIST digit image with its label.

    Args:
        image (numpy.ndarray): 1D array of pixel values (784 elements)
        label (int): The ground truth label (0-9) for the digit
        title (str): The title to display above the image

    Returns:
        None. Displays the image using matplotlib.

    Notes:
        The function assumes the input image is a 1D array of 784 elements,
        which it reshapes to 28x28 pixels for display.
    """
    # Reshape the image to 28x28 pixels
    image = image.reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.title(f'{title} - Label: {label}')
    plt.axis('off')
    plt.show()

def main():
    """
    Main function to run the MNIST visualization tool.

    This function:
    1. Parses command line arguments for dataset type and sample index
    2. Loads the MNIST dataset from a pickle file
    3. Retrieves and displays the requested sample

    Command line arguments:
        dataset_type: Either 'train' or 'test'
        index: Integer index of the sample to display

    Example usage:
        python script.py train 42  # Display the 42nd training sample
        python script.py test 10   # Display the 10th test sample
    """
    parser = argparse.ArgumentParser(description='Test MnistData class.')
    parser.add_argument('dataset_type', type=str, choices=['train', 'test'], 
                       help='The type of dataset (train or test)')
    parser.add_argument('index', type=int, 
                       help='The index of the sample to retrieve')
    args = parser.parse_args()

    dataset_pkl_path = 'dataset/mnist.pkl'  # Update this path
    mnist_data = MnistData(dataset_pkl_path)
    mnist_data.load()

    image, label = mnist_data.get_sample(args.dataset_type, args.index)
    visualize_sample(image, label, f'{args.dataset_type.capitalize()} Sample {args.index}')
    print(f'Label: {label}')

if __name__ == "__main__":
    main()