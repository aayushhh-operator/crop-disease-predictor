import streamlit as st
import os
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D
import requests
from PIL import Image

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

# Upload and predict logic
def predict_disease(image_file):
    # Preprocess the image and get the features
    image_path = f"uploads/{image_file.name}"
    with open(image_path, "wb") as f:
        f.write(image_file.getbuffer())
    
    features = preprocess_image(image_path)
    
    # Predict using the Random Forest model
    predicted_class = rf_model.predict(features)
    predicted_disease = label_encoder.inverse_transform(predicted_class)
    
    return predicted_disease[0]

# Streamlit Interface
st.title("Crop Disease Prediction App")
st.write("Upload a crop image to get the disease prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make Prediction
    disease = predict_disease(uploaded_file)
    st.write(f"Predicted Disease: {disease}")

# API Endpoint for external use
@st.cache
def predict_image_from_api(image_file):
    api_url = 'http://localhost:5000/api/predict_disease'
    files = {'file': image_file}
    response = requests.post(api_url, files=files)

    if response.status_code == 200:
        return response.json()['predicted_disease']
    else:
        return "Error: Unable to predict disease."

# Test API functionality
if st.button('Test API with uploaded image'):
    disease_from_api = predict_image_from_api(uploaded_file)
    st.write(f"Predicted Disease via API: {disease_from_api}")
