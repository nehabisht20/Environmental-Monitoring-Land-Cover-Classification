import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import requests
import tempfile
import os

# Load model from Google Drive link
@st.cache_resource
def load_trained_model():
    url = "https://drive.google.com/uc?export=download&id=1Y5bmVguQu7-RcIx0LGnKlhbtM5kN9EM9"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
        tmp_path = tmp_file.name
        response = requests.get(url)
        tmp_file.write(response.content)
    model = load_model(tmp_path)
    return model

model = load_trained_model()
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# App UI
st.set_page_config(page_title="Land Cover Classification", layout="centered")
st.title("🌍 Environmental Monitoring: Land Cover Classification")
st.write("Upload a satellite image to classify it as Cloudy, Desert, Green Area, or Water.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_resized = image.resize((255, 255))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"✅ Predicted Class: **{predicted_class}**")
