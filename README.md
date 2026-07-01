# i2i Academy - Computer Vision - Assignment 3

This repository contains the real-time Finger Counter application developed using modern AI frameworks as part of the i2i Academy Training Program.

## Project Overview
The application captures live video from the webcam, utilizes a state-of-the-art machine learning model to track human hand landmarks, and accurately counts the number of extended fingers in real-time. It dynamically supports both left and right hands (up to 10 fingers simultaneously) by adapting to mirror flips.

## Technical Architecture & Core Dependencies
- **Python 3** (Fully compatible with Python 3.16+)
- **OpenCV (`opencv-python`)**: For high-performance video capture, frame reflection (`cv2.flip`), and rendering dynamic text overlays.
- **Google MediaPipe (`mediapipe`)**: Powered by the advanced **MediaPipe Tasks API (v0.10.35+)** for on-device machine learning inference.

## The AI Model (`.task`)
The project relies on the pre-trained `hand_landmarker.task` bundle (a compiled TensorFlow Lite model). This architectural decoupling separates the high-fidelity neural network weights (3 MB binary) from the core Python engine, allowing for future seamless model swaps without upgrading library dependencies.

## Algorithmic Logic
- **Mirror Mitigation:** The raw camera stream is mirrored via `cv2.flip(frame, 1)` for a natural UX.
- **Handedness Tracking:** MediaPipe identifies the hand type (`Left`/`Right`).
- **Finger Kinematics:** - For the 4 main fingers, extension is validated by checking if the tip landmark's $Y$-coordinate is numerically lower than its lower joint PIP (Proximal Interphalangeal) joint.
  - For the thumb, horizontal $X$-coordinates are monitored. Due to mirror reflection, a true right hand's thumb extends toward decreasing $X$ space, while a left hand's thumb extends toward increasing $X$ space.

## How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/aytugotmar/i2i-Academy-ComputerVision-3.git](https://github.com/aytugotmar/i2i-Academy-ComputerVision-3.git)
