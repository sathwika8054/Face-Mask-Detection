# Face Mask Detection System

## Project Overview
This project is a web-based Face Mask Detection System built using Python, Flask, OpenCV, and TensorFlow. It uses a deep learning model to classify detected faces as wearing a mask or not wearing a mask.

## Technologies Used
- Python
- Flask
- OpenCV
- TensorFlow and Keras
- HTML, CSS, JavaScript
- Gunicorn
- Render for deployment

## Features
- Real-time camera access through the browser
- Face detection using OpenCV's deep learning module
- Mask classification using a trained deep learning model
- Web-based user interface

## Project Structure
- `app.py` — Flask application and detection logic
- `requirements.txt` — Python dependencies
- `templates/` — HTML templates
- `static/` — CSS and JavaScript files
- `face_detector/` — Face detection model files
- `mask_detector.h5` — Trained mask classification model

## Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/sathwika8054/Face-Mask-Detection.git
   ```
2. Open the project folder:
   ```bash
   cd Face-Mask-Detection
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the application:
   ```bash
   python app.py
   ```

## Note
Camera access requires browser permission and HTTPS when accessed publicly. Detection accuracy depends on the trained model and the quality of the camera input.

## Author
Face Mask Detection Project
