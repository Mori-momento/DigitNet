import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import base64
import re

# Suppress TensorFlow INFO/WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

MODEL_PATH = 'mnist_cnn_best.h5'
if os.path.exists(MODEL_PATH):
     print(f"Loading existing model from {MODEL_PATH}...")
     model = tf.keras.models.load_model(MODEL_PATH)
else:
     print("No existing model found. Training a new one...")
     model = create_and_train_model()
     print(f"Saving trained model to {MODEL_PATH}...")
     model.save(MODEL_PATH) # Save the trained model

# --- Image Preprocessing ---
def preprocess_image(img_data):
    # Decode base64
    # Remove header: "data:image/png;base64,"
    img_str = re.search(r'base64,(.*)', img_data).group(1)
    nparr = np.frombuffer(base64.b64decode(img_str), np.uint8)

    # Read image with OpenCV
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE) # Read as grayscale directly

    if img is None:
        raise ValueError("Could not decode image")

    # Resize to 28x28
    img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Invert colors (MNIST expects white digit on black background)
    # Canvas usually draws black on white
    img_inverted = cv2.bitwise_not(img_resized)

    # Thresholding (optional but often helpful for canvas drawings)
    # Adjust threshold value if needed
    _, img_thresh = cv2.threshold(img_inverted, 50, 255, cv2.THRESH_BINARY)

    # Normalize and reshape for model
    img_final = img_thresh.astype('float32') / 255.0
    img_final = np.reshape(img_final, (1, 28, 28, 1)) # Add batch and channel dimensions

    return img_final

# --- Flask Routes ---
@app.route('/')
def index():
    # Serve the HTML page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data found'}), 400

        img_data = data['image']
        processed_img = preprocess_image(img_data)

        # Make prediction - returns probabilities for each class (0-9)
        probabilities = model.predict(processed_img)[0] # Get the probabilities array
        predicted_digit = int(np.argmax(probabilities)) # Get the index of the highest probability
        probabilities_list = probabilities.tolist() # Convert numpy array to Python list

        # Return both prediction and probabilities
        return jsonify({'prediction': predicted_digit, 'probabilities': probabilities_list})

    except ValueError as ve:
         print(f"Preprocessing Error: {ve}")
         return jsonify({'error': f'Image processing error: {ve}'}), 400
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': f'An error occurred: {e}'}), 500

if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible on the network if needed
    app.run(host='0.0.0.0', port=5000, debug=True) # Turn debug=False for production