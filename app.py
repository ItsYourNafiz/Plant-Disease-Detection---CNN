from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array
import cv2

# Load the trained model
model = load_model('/content/drive/My Drive/Plant_images_pianalytix/plant_disease_model.h5')

# Define class labels
all_labels = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

st.title("🌿 Plant Disease Detection")

# Webcam option
run_webcam = st.checkbox('Run Webcam')

if run_webcam:
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        # To read image file buffer as a PIL Image:
        img = Image.open(img_file_buffer)
        img = img.resize((256, 256))  # Resize image
        st.image(img, caption='Webcam Image.', use_column_width=True)
        image_array = img_to_array(img) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        prediction = model.predict(image_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class = all_labels[predicted_class_index]
        confidence = prediction[0][predicted_class_index]
        st.write(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}")

# Upload image option
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)

    # Preprocess the image
    image = image.resize((256, 256))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    prediction = model.predict(image_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = all_labels[predicted_class_index]
    confidence = prediction[0][predicted_class_index]

    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
