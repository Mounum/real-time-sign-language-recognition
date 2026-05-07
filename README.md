# Real-Time Sign Language Recognition System

> A real-time sign language recognition system using MediaPipe and MobileNetV2 for accurate hand gesture detection and classification.

---

## Overview

This project focuses on building a real-time sign language recognition system that helps reduce communication barriers for people with hearing and speech impairments.

The system captures live webcam input, detects hand landmarks using MediaPipe, and classifies gestures using a deep learning model based on MobileNetV2.

The application is designed to provide fast and accurate gesture recognition with low latency, making it suitable for real-time communication systems.

---

## Features

- Real-time webcam-based gesture recognition
- Hand landmark extraction using MediaPipe
- Deep learning classification using MobileNetV2
- Fast and low-latency predictions
- Multiple sign gesture recognition
- User-friendly interface
- Real-time prediction display

---

## Technologies Used

- Python
- TensorFlow
- MobileNetV2
- MediaPipe
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

## System Architecture

The workflow of the system is:

1. Capture live webcam video
2. Detect hand landmarks using MediaPipe
3. Extract 21 hand keypoints
4. Preprocess and normalize landmark data
5. Classify gestures using MobileNetV2
6. Display predicted sign in real time

---

## Project Structure

```bash
real-time-sign-language-recognition/
│
├── src/
│   ├── train_clean.py
│   ├── webcam_test.py
│   ├── evaluate_model.py
│
├── screenshots/
│
├── model/
│
├── dataset/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Mounum/real-time-sign-language-recognition.git
```

Go to the project folder:

```bash
cd real-time-sign-language-recognition
```

Install required libraries:

```bash
pip install -r requirements.txt
```

---

## Run the Project

Run the real-time prediction system:

```bash
python src/webcam_test.py
```

Train the model:

```bash
python src/train_clean.py
```

Evaluate model performance:

```bash
python src/evaluate_model.py
```

---

## Dataset

The dataset contains multiple sign language gesture classes collected from different users under varying lighting conditions, backgrounds, and camera angles.

The dataset is used to improve:
- robustness
- gesture diversity
- real-time accuracy

---

## Results

The proposed system achieved:
- high gesture classification accuracy
- low response time
- smooth real-time predictions

The model performs efficiently in recognizing multiple hand gestures using live webcam input.

---

## Screenshots


- User interface
- Gesture prediction output
- Accuracy graph
- Confusion matrix
- Real-time detection window

## Screenshots

### User Interface
![UI Demo](screenshots/ui-demo.png)

### Prediction Output
![Prediction Output](screenshots/prediction-output.png)

### Confusion Matrix
![Confusion Matrix](screenshots/confusion-matrix.png)

### Accuracy Graph
![Accuracy Graph](screenshots/accuracy-graph.png)

## Future Improvements

- Text-to-speech conversion
- Sentence generation from gestures
- Support for more sign languages
- Cloud deployment
- Mobile application support

---

## Applications

- Communication assistance for hearing-impaired individuals
- Educational tools
- Human-computer interaction
- Gesture-controlled systems

---

## Contributors

- Mounusha Medisetti
- Parthiv K
- Dr. Muthu Kumaran AMJ

---

