import cv2
import numpy as np
import tensorflow as tf

# Function to load class names from labels.txt
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

# Load the model path
model = tf.keras.models.load_model(r"C:\Users\shubdosh\Desktop\ece5831\05\model\keras_model.h5")
"""
Loads the pre-trained Keras model from the specified file path.

Returns:
    model (tensorflow.python.keras.engine.functional.Functional): The loaded Keras model for making predictions.
"""

# Load the class names from the labels.txt file
class_names = load_labels(r"C:\Users\shubdosh\Desktop\ece5831\05\model\labels.txt")
"""
Loads the class names from a specified labels.txt file.

Returns:
    class_names (list): A list of class names corresponding to the model's output.
"""

# Initialize the webcam
cap = cv2.VideoCapture(0)
"""
Initializes the webcam for video capture. The argument '0' selects the default webcam.

Returns:
    cap (cv2.VideoCapture): A video capture object to capture frames from the webcam.
"""

# Get the width and height of the frame
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
"""
Retrieves the width and height of the video frame from the webcam.

Returns:
    frame_width (int): The width of the video frame.
    frame_height (int): The height of the video frame.
"""

# Define the codec and create a VideoWriter object to save the video
output_file = r"C:\Users\shubdosh\Desktop\ece5831\05\output.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file, fourcc, 20.0, (frame_width, frame_height))
"""
Creates a VideoWriter object to save the webcam video to a file.

Args:
    output_file (str): Path where the output video file will be saved.
    fourcc (str): Codec used to compress the frames (XVID in this case).
    20.0 (float): Frames per second for the output video.
    (frame_width, frame_height) (tuple): Size of the video frame.
"""

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    """
    Reads a frame from the webcam video feed.

    Returns:
        ret (bool): Whether the frame was successfully captured.
        frame (numpy.ndarray): The captured frame as a numpy array.
    """
    
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the image for model prediction (resize and normalize)
    img = cv2.resize(frame, (224, 224))  # Resize to model's input size
    """
    Resizes the captured frame to 224x224 pixels, which is the input size expected by the model.

    Returns:
        img (numpy.ndarray): The resized image.
    """
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize image
    """
    Normalizes the pixel values of the image to the range [0, 1].

    Returns:
        img (numpy.ndarray): The normalized image.
    """
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    """
    Adds a batch dimension to the image to make it compatible with the model input.

    Returns:
        img (numpy.ndarray): The preprocessed image ready for prediction.
    """

    # Predict the class
    predictions = model.predict(img)
    """
    Runs the preprocessed image through the model to obtain class predictions.

    Returns:
        predictions (numpy.ndarray): The model's prediction probabilities for each class.
    """
    class_idx = np.argmax(predictions)
    """
    Retrieves the index of the class with the highest probability from the model's predictions.

    Returns:
        class_idx (int): Index of the predicted class.
    """
    prediction_label = class_names[class_idx]
    """
    Maps the predicted class index to the corresponding class name.

    Returns:
        prediction_label (str): The predicted class label.
    """
    
    # Display the resulting frame with prediction
    cv2.putText(frame, f'Prediction: {prediction_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    """
    Adds text displaying the predicted class label onto the video frame.

    Args:
        frame (numpy.ndarray): The video frame on which the text will be drawn.
        f'Prediction: {prediction_label}' (str): The text to be drawn on the frame.
        (10, 30) (tuple): Position of the text on the frame.
        cv2.FONT_HERSHEY_SIMPLEX (int): Font type for the text.
        1 (float): Font size.
        (255, 0, 0) (tuple): Text color in BGR format (here, blue).
        2 (int): Thickness of the text.
    """
    cv2.imshow('Rock Paper Scissors', frame)
    """
    Displays the video frame with the predicted class label in a window.

    Args:
        'Rock Paper Scissors' (str): The window title.
        frame (numpy.ndarray): The video frame to be displayed.
    """

    # Save the frame to the output video file
    out.write(frame)
    """
    Writes the current video frame with the predicted label to the output video file.

    Args:
        frame (numpy.ndarray): The video frame to be saved to the file.
    """

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        """
        Waits for the 'q' key to be pressed to break the loop and stop capturing video.

        Returns:
            bool: True if 'q' is pressed, otherwise continues the loop.
        """
        break

# When everything is done, release the capture and video writer
cap.release()
"""
Releases the webcam resource.
"""
out.release()  # Save the video
"""
Releases the VideoWriter resource and saves the output video to the file.
"""
cv2.destroyAllWindows()
"""
Closes all OpenCV windows.
"""

print(f"Video saved as {output_file}")
"""
Prints a confirmation message with the path of the saved video file.
"""
