# Drowsiness Detection System

This project is a real-time drowsiness detection system that uses a webcam to monitor a user's eye aspect ratio (EAR). If the EAR falls below a certain threshold for a specified number of frames, the system triggers an alarm to alert the user.

## Features
- Real-time detection of drowsiness using a webcam.
- Uses facial landmarks to calculate the Eye Aspect Ratio (EAR).
- Plays an alarm sound (`alarm.wav`) when drowsiness is detected.

## Requirements
- Python 3.6 to 3.10 (Python 3.12 is not supported by `dlib`).
- A webcam for real-time video capture.

## Dependencies
Install the required Python libraries using the following command:
```bash
pip install scipy imutils pygame opencv-python opencv-python-headless dlib numpy