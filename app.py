from flask import Flask, request, render_template
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import io
from PIL import Image

# Create a Flask application instance
app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')

# Define a route for the homepage
@app.route("/")
def home():
    return render_template("index.html")  # Render the index.html template

# Define a route for making predictions
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return render_template("index.html", prediction="No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template("index.html", prediction="No file selected")

    if file:
        # Read and preprocess the image
        image = Image.open(io.BytesIO(file.read()))
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to the expected input shape
        image_array = np.array(image) / 255.0  # Normalize
        image_array = np.reshape(image_array, (1, 28, 28))  # Reshape for model

        # Make a prediction
        prediction = model.predict(image_array)
        predicted_label = np.argmax(prediction)

        return render_template("index.html", prediction=f'The digit is recognized as {predicted_label}')

# Start the Flask application
if __name__ == "__main__":
    app.run(debug=True)  # Run the application in debug mode for development
