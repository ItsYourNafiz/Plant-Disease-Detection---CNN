import streamlit as st
import numpy as np
from PIL import Image
import requests
import tensorflow as tf
from io import BytesIO
from google_drive_downloader import GoogleDriveDownloader as gdd

# Google Drive File IDs
json_file_id = "1JjBG0rkLiiIjpPIIJVFemWJT9crSQ-4M"
weights_file_id = "1k6Yjhfjrr53311iQf4iz4eWtfU_mjZHg"
h5_file_id = "1qngkqm4LAwvcM_qbrgBfLMT0yOzcl7TD"

# File paths
json_file_path = "model.json"
weights_file_path = "model_weights.h5"
h5_file_path = "model.h5"

# Download the files
@st.cache_resource
def download_model_files():
    gdd.download_file_from_google_drive(file_id=json_file_id, dest_path=json_file_path, unzip=False)
    gdd.download_file_from_google_drive(file_id=weights_file_id, dest_path=weights_file_path, unzip=False)
    gdd.download_file_from_google_drive(file_id=h5_file_id, dest_path=h5_file_path, unzip=False)
    return json_file_path, weights_file_path, h5_file_path

# Load model
@st.cache_resource
def load_model(json_path, weights_path):
    with open(json_path, 'r') as json_file:
        model_json = json_file.read()
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(weights_path)
    return model

# Preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust size according to model input
    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0  # Normalize the image
    return image_array

# Main application
st.title(":herb: Plant Disease Detection")

st.write("Upload an image of a plant leaf to detect potential diseases.")

# Download and load model
st.write("Loading model...")
json_path, weights_path, h5_path = download_model_files()
model = load_model(json_path, weights_path)
st.write("Model loaded successfully!")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    st.write("Processing image...")
    processed_image = preprocess_image(image)

    # Make prediction
    st.write("Making prediction...")
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Map predictions to classes (update according to your model)
    class_labels = {
        0: "Healthy",
        1: "Bacterial Spot",
        2: "Early Blight",
        3: "Late Blight",
        4: "Leaf Mold",
        5: "Septoria Leaf Spot",
        6: "Spider Mites",
        7: "Target Spot",
        8: "Yellow Leaf Curl Virus",
        9: "Mosaic Virus",
        10: "Gray Spot"
    }

    result = class_labels.get(predicted_class, "Unknown Disease")

    # Display result
    st.write(f"Prediction: {result}")
