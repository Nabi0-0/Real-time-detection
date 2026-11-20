# Vision Intelligence Lab

A Python computer vision project exploring real-time webcam applications using OpenCV, MediaPipe, and YOLOv3.

## Features

- Face detection
- Facial landmark detection using MediaPipe FaceMesh
- YOLOv3 object detection
- Approximate face-distance estimation
- Webcam snapshots and video recording
- Optional email alerts

## Technology Stack

- Python 3.12
- OpenCV
- MediaPipe
- NumPy
- YOLOv3

## Installation

Create and activate a virtual environment:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

Install dependencies:

    python -m pip install -r requirements.txt

## YOLOv3 Weights

Download the official weights:

https://pjreddie.com/media/files/yolov3.weights

Place the file at:

    Object_detection/yolov3.weights

The weights are excluded from Git because of their size.

The configuration file is named yolov3-320.cfg, but contains the full YOLOv3 configuration for 608 x 608 input.

## Running the Project

Run commands from the project root:

    python .\final_file.py
    python .\Object_detection\objectdetect.py
    python .\Face_detection\FaceDetectionBasics.py
    python .\Face_detection\FaceDetectionModule.py
    python ".\extra files\recording.py"
    python ".\extra files\snapshotfile.py"
    python ".\extra files\alert.py"

## Email Alerts

Email alerts require Gmail credentials supplied through environment variables:

    $env:ALERT_EMAIL = "your-email@gmail.com"
    $env:ALERT_PASSWORD = "your-gmail-app-password"
    python ".\extra files\alert.py"

Use a Gmail App Password where supported. Never hardcode credentials or commit them to Git.

## Limitations

- A webcam is required for real-time applications.
- YOLOv3 object detection requires the downloaded weights.
- Face-distance estimation is approximate and uncalibrated.
- Email alerts require valid credentials and an SMTP connection.
- Performance depends on lighting, camera quality, and hardware.

## Project Status

Originally developed as a college project, this repository is being incrementally improved with a focus on code quality, reproducibility, machine learning experimentation, and computer vision engineering.
