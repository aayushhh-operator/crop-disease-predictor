<<<<<<< HEAD
import streamlit as st
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D
import os

# Load the Random Forest model and label encoder
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

# Streamlit UI
st.title('Crop Disease Detection')

# File uploader for image input
uploaded_file = st.file_uploader("Choose a crop image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Save the uploaded file to the static/uploads directory
    uploads_dir = os.path.join('static', 'uploads')
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

    # Define the path to save the uploaded file
    image_path = os.path.join(uploads_dir, uploaded_file.name)

    # Save the file
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Display the image
    st.image(image_path, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    features = preprocess_image(image_path)

    # Predict using the Random Forest model
    predicted_class = rf_model.predict(features)

    # Decode the predicted class label using the label encoder
    predicted_disease = label_encoder.inverse_transform(predicted_class)

    # Display the result
=======
import streamlit as st
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D
import os

# Load the Random Forest model and label encoder
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

# Streamlit UI
st.title('Crop Disease Detection')

# File uploader for image input
uploaded_file = st.file_uploader("Choose a crop image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Save the uploaded file to the static/uploads directory
    uploads_dir = os.path.join('static', 'uploads')
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

    # Define the path to save the uploaded file
    image_path = os.path.join(uploads_dir, uploaded_file.name)

    # Save the file
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Display the image
    st.image(image_path, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    features = preprocess_image(image_path)

    # Predict using the Random Forest model
    predicted_class = rf_model.predict(features)

    # Decode the predicted class label using the label encoder
    predicted_disease = label_encoder.inverse_transform(predicted_class)

    # Display the result
>>>>>>> b4e531da0381eb5cedad8d8bae1984235d3578ee
    st.write(f"Predicted Disease: {predicted_disease[0]}")