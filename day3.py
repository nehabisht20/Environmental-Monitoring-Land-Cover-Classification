import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import gdown
import os

# Download model from Google Drive using gdown
@st.cache_resource
def load_model_from_drive():
    url = "https://drive.google.com/uc?export=download&id=1p9pqC-Ba4aKdNcQploHjnaCVip5J07qe"
    output_path = "Modelenv.v1.h5"
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)
    return load_model(output_path)

# Load model
model = load_model_from_drive()

# Labels
labels = {
    0: "Cloudy",
    1: "Desert",
    2: "Green_Area",
    3: "Water"
}

st.set_page_config(page_title="🌍 Satellite Classifier", layout="centered")
st.title("🛰 Satellite Image Classifier")
st.markdown("Upload a satellite image and let AI predict the terrain!")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((256, 256))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    prediction = model.predict(img_array)
    predicted_class = labels[np.argmax(prediction)]

    st.success(f"🌟 Prediction: *{predicted_class}*")
    st.bar_chart(prediction[0])
else:
    st.info("Please upload a satellite image to get started.")

st.caption("🔍 Model loaded from Google Drive using gdown.")
