Here's a sample `README.md` for your project based on the three codes you provided:

--- 
Youtube link - https://youtu.be/DCcP2pdPuG4
# Rock-Paper-Scissors 

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Classifying a Single Image](#classifying-a-single-image)
  - [Real-time Webcam Classification](#real-time-webcam-classification)
- [Files Overview](#files-overview)
- [License](#license)

---

## Requirements

Libraries installed:

- `numpy`
- `opencv-python`
- `tensorflow`
- `matplotlib`

You can install the dependencies using:
```bash
pip install numpy opencv-python tensorflow matplotlib
```

## Installation
   - Model file: `keras_model.h5`
   - Label file: `labels.txt`

3. Update the paths in the code for both the model and label files as per your system configuration.

```python
MODEL_PATH = r"C:\path\to\model\keras_model.h5"
LABELS_PATH = r"C:\path\to\model\labels.txt"
```

4. If you plan to run real-time classification, ensure you have a working webcam.

## Usage

### Classifying a Single Image

You can classify a single image by running rock-paper-scissor.py

```bash
python classify_image.py <image_path>
```

For example:

```bash
python classify_image.py C:\path\to\your\image.jpg
```

This script will:
- Load the specified image.
- Preprocess it (resize to 224x224 and normalize pixel values).
- Use the pre-trained model to predict the class (Rock, Paper, Scissors).
- Display the image with the predicted class and confidence score.

### Real-time Webcam Classification

For real-time hand gesture classification using your webcam, run:

```bash
python rock-paper-scissor-live.py
```

This script will:
- Open the webcam and capture frames in real-time.
- Preprocess each frame (resize to 224x224 and normalize).
- Use the pre-trained model to predict the gesture.
- Display the webcam feed with the predicted class overlaid.
- Save the processed video to an output file (`output.avi`).

### Video Saving

The webcam-based classifier also saves the video stream to an output file (`output.avi`) while displaying the classification results in real-time. You can modify the output file path in the code:

```python
output_file = r"C:\path\to\save\output.avi"
Youtube link - https://youtu.be/DCcP2pdPuG4
```

## Files Overview

- **`rock-paper-scissor.py`**: This script loads a single image from the specified path, processes it, and predicts the class (Rock, Paper, Scissors) using a pre-trained model. The prediction and confidence score are displayed along with the image.

 **`teachable.ipynb`**: This ipynb loads a single image from the specified path, processes it, and predicts the class (Rock, Paper, Scissors) using a pre-trained model. The prediction and confidence score are displayed after the code cell is run.

- **`rock-paper-scissor-live.py`**: This script captures video from the webcam in real-time, processes each frame, and predicts the class (Rock, Paper, Scissors). The predicted class is displayed on the video feed, and the video is saved to an output file.
  
- **`labels.txt`**: This file contains the class names (e.g., Rock, Paper, Scissors) that correspond to the model's output classes.
  
- **`keras_model.h5`**: Pre-trained TensorFlow Keras model for classifying hand gestures.

### Sample Code Structure

```
.
├── rock-paper-scissor.py          # Script to classify a single image
├── rock-paper-scissor-live.py    # Script to classify gestures from webcam feed
├── teachable.ipynb            # ipynb image for testing implementation of rock-paper-scissor code 
├── model
│   ├── keras_model.h5         # Pre-trained model (download separately)
│   └── labels.txt             # Class names for the model (download separately)
├── output.avi                 # Saved video file from webcam classification
└── README.md                  # Project documentation
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Example Command

To classify an image:

```bash
python rock-paper-scissor.py C:\path\to\image.jpg
```

To classify real-time video feed:

```bash
python rock-paper-scissor-live.py
```
--- 
