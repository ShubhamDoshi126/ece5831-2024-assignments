import os
import pickle
import numpy as np

class MnistData:
    """
    A class for handling the MNIST dataset of handwritten digits.
    
    This class manages the downloading, loading, and preprocessing of the MNIST dataset.
    It provides functionality to access both training and test datasets, with images
    normalized to the range [0, 1].

    Attributes:
        dataset_pkl_path (str): Path to the pickle file containing the MNIST dataset
        dataset (dict): Dictionary containing the training and test datasets with keys:
            - 'train_images': Training images
            - 'train_labels': Training labels
            - 'test_images': Test images
            - 'test_labels': Test labels
    """

    def __init__(self, dataset_pkl_path):
        """
        Initialize the MnistData object.

        Args:
            dataset_pkl_path (str): Path where the MNIST dataset pickle file should be 
                stored or loaded from.

        Notes:
            If the pickle file exists at the specified path, it will be loaded.
            Otherwise, the dataset will be downloaded and created.
        """
        self.dataset_pkl_path = dataset_pkl_path
        self.dataset = {}
        self._download_all()
        if os.path.exists(f'{self.dataset_pkl_path}'):
            with open(f'{self.dataset_pkl_path}', 'rb') as f:
                print(f'Pickle: {self.dataset_pkl_path} already exists.')
                print('Loading...')
                self.dataset = pickle.load(f)
                print('Done.')
        else:
            self._create_dataset()

    def _download_all(self):
        """
        Download the MNIST dataset files if they don't exist locally.

        This is a private method that handles the downloading of raw MNIST data files.
        The actual implementation details are not provided in this skeleton.
        """
        pass

    def _create_dataset(self):
        """
        Create and save the MNIST dataset in pickle format.

        This is a private method that processes the raw MNIST files and creates
        a structured dataset. The actual implementation details are not provided
        in this skeleton.
        """
        pass

    def load(self):
        """
        Normalize the image datasets to the range [0, 1].

        This method processes both training and test images by:
        1. Converting the data type to float32
        2. Scaling pixel values from [0, 255] to [0, 1]

        The normalization is performed in-place on the dataset dictionary.
        """
        # normalize image datasets
        for key in ('train_images', 'test_images'):
            self.dataset[key] = self.dataset[key].astype(np.float32)
            self.dataset[key] /= 255.0

    def get_sample(self, dataset_type, index):
        """
        Retrieve a single sample (image and label) from the dataset.

        Args:
            dataset_type (str): The type of dataset ('train' or 'test')
            index (int): The index of the sample to retrieve

        Returns:
            tuple: A tuple containing:
                - image (numpy.ndarray): A flattened array of 784 pixel values
                - label (int): The corresponding label for the image

        Raises:
            ValueError: If dataset_type is not 'train' or 'test'
            ValueError: If the requested dataset type is not found
        """
        if dataset_type not in ['train', 'test']:
            raise ValueError("dataset_type must be 'train' or 'test'")

        images_key = f'{dataset_type}_images'
        labels_key = f'{dataset_type}_labels'

        if images_key not in self.dataset or labels_key not in self.dataset:
            raise ValueError(f"Dataset type '{dataset_type}' not found in the dataset")

        image = self.dataset[images_key][index]
        label = self.dataset[labels_key][index]

        return image, label

    def load_data(self):
        """
        Get the complete MNIST dataset split into training and test sets.

        Returns:
            tuple: A tuple containing two tuples:
                - ((train_images, train_labels), (test_images, test_labels))
                where:
                - train_images: numpy.ndarray of shape (n_samples, 784) containing flattened training images
                - train_labels: numpy.ndarray of shape (n_samples,) containing training labels
                - test_images: numpy.ndarray of shape (n_samples, 784) containing flattened test images
                - test_labels: numpy.ndarray of shape (n_samples,) containing test labels

        Notes:
            - Images are flattened to 784 bytes (28x28 pixels)
            - Labels are one-hot-encoded arrays
        """
        return (self.dataset['train_images'], self.dataset['train_labels']), (self.dataset['test_images'], self.dataset['test_labels'])

if __name__ == "__main__":
    print("MnistData class is to load MNIST datasets.")