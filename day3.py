import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import gdown
import os

# Load model from Google Drive
@st.cache_resource
def load_trained_model():
    model_path = "Modelenv.v1.h5"
    if not os.path.exists(model_path):
        file_id = "1Y5bmVguQu7-RcIx0LGnKlhbtM5kN9EM9"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    model = load_model(model_path)
    return model

model = load_trained_model()
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# Streamlit UI
st.set_page_config(page_title="Land Cover Classification", layout="centered")
st.title("🌍 Land Cover Classification Using Satellite Images")
st.write("Upload a satellite image to classify it as **Cloudy**, **Desert**, **Green Area**, or **Water**.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_resized = image.resize((255, 255))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"✅ Predicted Class: **{predicted_class}**")
