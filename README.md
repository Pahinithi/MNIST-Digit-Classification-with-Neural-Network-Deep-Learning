# Digit Prediction with Neural Network Deep Learning

This project is a web application that allows users to upload an image of a handwritten digit and predict the digit using a trained neural network model. The application is built using Flask and TensorFlow.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Model](#model)
- [File Structure](#file-structure)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This application leverages a deep learning model trained on the MNIST dataset to recognize and predict handwritten digits (0-9). The application provides a user-friendly interface where users can upload an image of a digit, and the model will output the predicted digit.

## Features

- Upload an image of a handwritten digit.
- Real-time prediction of the digit.
- Responsive and modern UI with Bootstrap styling.
- Supports various image formats.

## Demo

You can view a live demo of the application [here](#).

## Installation

To run this application locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Pahinithi/MNIST-Digit-Classification-with-Neural-Network-Deep-Learning
   cd Digit-Prediction-Flask-App
   ```

2. **Create a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the trained model:**

   Download the `model.h5` file and place it in the root directory of the project.

5. **Run the Flask application:**

   ```bash
   python app.py
   ```

6. **Open your browser:**

   Visit `http://127.0.0.1:5000` to use the application.

## Usage

1. Upload an image of a digit using the "Choose Image" button.
2. Click the "Upload and Predict" button to see the predicted digit.
3. The application will display the predicted digit below the upload form.

## Technologies Used

- **Flask:** Web framework for Python.
- **TensorFlow & Keras:** For building and training the neural network.
- **OpenCV & PIL:** For image processing.
- **Bootstrap:** For responsive and modern UI design.
- **Matplotlib & Seaborn:** For visualizing data.

## Model

The model is a simple neural network with the following architecture:

- **Input Layer:** Flatten layer to convert 28x28 images to 1D vectors.
- **Hidden Layers:** Two dense layers with 50 neurons each and ReLU activation.
- **Output Layer:** Dense layer with 10 neurons and softmax activation for digit classification.

The model is trained on the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.

## File Structure

```
Digit-Prediction-Flask-App/
├── templates/
│   └── index.html        # HTML template for the application
├── app.py                # Main Flask application file
├── model.h5              # Trained model file
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## License

This project is licensed under the MIT License. 

## Acknowledgments

- Special thanks to the creators of the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for providing an excellent resource for digit recognition.
- Inspired by various machine learning tutorials and Flask documentation.
