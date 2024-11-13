import os
import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np
from mnist import Mnist

def prepare_image(image_path):
    """
    Prepare a custom MNIST-like image for prediction with black background and white digit.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Processed image ready for prediction (black background, white digit).
    """
    # Read image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image {image_path}")
    
    # Threshold the image to make it purely black and white
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Find the bounding box of the digit
    coords = cv2.findNonZero(255 - image)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Add padding around the digit
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2*padding)
        h = min(image.shape[0] - y, h + 2*padding)
        image = image[y:y+h, x:x+w]
    
    # Create a square image with padding (for centering)
    size = max(image.shape[0], image.shape[1])
    square_image = np.full((size, size), 0, dtype=np.uint8)  # Start with black (0)

    # Center the digit in the square image
    y_offset = (size - image.shape[0]) // 2
    x_offset = (size - image.shape[1]) // 2
    square_image[y_offset:y_offset+image.shape[0], 
                x_offset:x_offset+image.shape[1]] = image
    
    # Resize to 28x28 (standard MNIST image size)
    processed = cv2.resize(square_image, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Ensure the image has a black background with white digit
    # The result from `cv2.threshold` gives white digits on a black background,
    # so no need to invert after resizing.
    
    return processed

def main():
    """
    Main function to process an image, display it, and predict its label using the Mnist class.

    Usage:
        python module5-3.py <image_filename> <true_digit>
    """
    if len(sys.argv) != 3:
        print("Usage: python module5-3.py <image_filename> <true_digit>")
        sys.exit(1)
    
    image_filename = sys.argv[1]
    true_digit = int(sys.argv[2])
    
    # Set base directory to "Custom MNIST Samples" folder
    base_dir = os.path.join(os.path.dirname(__file__), "Custom MNIST Samples")
    # Construct the full image path based on expected digit
    image_path = os.path.join(base_dir, f"Digit {true_digit}", image_filename)
    
    print(f"Processing image: {image_path}")
    
    try:
        # Load original image for display (this is the raw image)
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Process image for prediction (with black background and white digit)
        processed = prepare_image(image_path)
        
        # Display both images
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original, cmap='gray')
        plt.title("Original Image")
        plt.axis('on')
        
        plt.subplot(1, 2, 2)
        plt.imshow(processed, cmap='gray')
        plt.title("Processed Image (28x28) with Black Background")
        plt.axis('on')
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()
        
        # Make prediction using the Mnist class
        mnist = Mnist()
        predicted_digit = mnist.predict(processed)
        
        # Compare predicted digit with true digit and print result
        if predicted_digit == true_digit:
            print(f"Success: Image {image_path} for digit {true_digit} is recognized as {predicted_digit}")
        else:
            print(f"Fail: Image {image_path} for digit {true_digit} is recognized as {predicted_digit}")
            
    except Exception as e:
        print(f"Error processing image: {e}")
        raise

if __name__ == "__main__":
    main()
