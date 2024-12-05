from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model and label encoder
rf_model = joblib.load('models/crop_disease_model.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# Load the pre-trained VGG16 model (without the top fully connected layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
model = Model(inputs=base_model.input, outputs=x)

# Function to preprocess the image
def preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image)
    features = features.flatten()
    features = np.expand_dims(features, axis=0)
    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Create the uploads directory if it doesn't exist
        uploads_dir = os.path.join('static', 'uploads')
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)

        # Define the path to save the uploaded file
        image_path = os.path.join(uploads_dir, secure_filename(file.filename))

        # Save the file
        file.save(image_path)

        # Preprocess the image
        features = preprocess_image(image_path)

        # Predict using the Random Forest model
        predicted_class = rf_model.predict(features)

        # Decode the predicted class label using the label encoder
        predicted_disease = label_encoder.inverse_transform(predicted_class)

        return jsonify({'predicted_disease': predicted_disease[0]})

# External API endpoint to accept image and return predicted disease
@app.route('/api/predict_disease', methods=['POST'])
def api_predict_disease():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Create the uploads directory if it doesn't exist
        uploads_dir = os.path.join('static', 'uploads')
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)

        # Define the path to save the uploaded file
        image_path = os.path.join(uploads_dir, secure_filename(file.filename))

        # Save the file
        file.save(image_path)

        # Preprocess the image
        features = preprocess_image(image_path)

        # Predict using the Random Forest model
        predicted_class = rf_model.predict(features)

        # Decode the predicted class label using the label encoder
        predicted_disease = label_encoder.inverse_transform(predicted_class)

        # Return the predicted disease
        return jsonify({'predicted_disease': predicted_disease[0]}), 200

if __name__ == "__main__":
    app.run(debug=True)
