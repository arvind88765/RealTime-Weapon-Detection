# Real-Time Weapon Detection Using YOLOv3

This project implements **real-time weapon detection** using the **YOLOv3 (You Only Look Once)** object detection model. It supports detection from images, videos, or real-time webcam input.

## Overview

The project leverages **YOLOv3**, a powerful deep learning algorithm for real-time object detection. It has been fine-tuned to detect specific objects (in this case, weapons) from images or video streams. The system allows users to choose from different modes, including processing an image, video, or real-time detection via webcam.

### Features:
- **Real-Time Detection:** Detect weapons in real-time via webcam.
- **Image and Video Detection:** Detect weapons in static images and videos.
- **Custom YOLOv3 Model:** The model is specifically trained to detect weapons (configured for the "Arm" class).
- **Interactive Menu:** Provides an easy-to-use interface for users to choose between image, video, or webcam-based detection.

## Project Structure

- **weapon_detection.py:** Main script for performing weapon detection, with an interactive menu for user input.
- **yolov3_testing.cfg:** YOLOv3 configuration file for the custom-trained model.
- **yolov3_training_2000.weights:** Pre-trained weights for the YOLOv3 model. [Download the weights file here](https://drive.google.com/file/d/1N5DqF6SJ_dXlT_-w0tb-_LFFS5uoH7nF/view?usp=sharing).
- **README.md:** Project description and setup instructions.

## Prerequisites

Before running the code, you need to install the required dependencies. The list includes:

- Python 3.x
- OpenCV
- NumPy
- PyTorch (or TensorFlow, depending on the implementation)

To install the required dependencies, you can create a virtual environment and install them using `pip`:


# Create a virtual environment
```
python -m venv env
```

# Activate the environment
# Windows:
```
env\Scripts\activate
```
# Linux/macOS:
```
source env/bin/activate
```

# Install dependencies
```
pip install opencv-python numpy torch
```



Setup Instructions
Clone the Repository:

First, clone this repository to your local machine:
```
git clone https://github.com/arvind88765/RealTime-Weapon-Detection.git
```
```
cd RealTime-Weapon-Detection
```

## Download Weights File:

- Since the YOLOv3 weights file is large, we host it on Google Drive. Download the file from the link below and place it in the project directory:

- Download YOLOv3 Weights File

- Make sure the file is named yolov3_training_2000.weights and placed in the project root directory.

- Make sure to place the images and videos you want to test in this folder.

- Edit the Configuration (if necessary):

- If needed, you can modify the yolov3_testing.cfg file to adjust parameters like confidence thresholds, input image sizes, etc.

## Running the Weapon Detection

- Once everything is set up, you can run the weapon detection script by executing the following command:
```
python weapon_detection.py
```

- Interactive Menu:
When you run the script, you’ll see a menu with the following options:

- Image Detection – Detect weapons in an image.
- Video Detection – Detect weapons in a video file.
- Real-time Webcam Detection – Detect weapons from your webcam feed.

## Contributing
If you'd like to contribute to this project, feel free to open an issue or create a pull request. All contributions are welcome!

