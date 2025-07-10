import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import gdown
import os

# Constants
MODEL_URL = "https://drive.google.com/uc?id=1Y5bmVguQu7-RcIx0LGnKlhbtM5kN9EM9"
MODEL_PATH = "Modelenv.v1.h5"
CLASS_NAMES = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# Load model safely from Drive
@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000:  # <10 KB likely HTML file
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
    return load_model(MODEL_PATH)

model = load_model_from_drive()

# Streamlit UI
st.set_page_config(page_title="Land Cover Classifier", layout="centered")
st.title("🌍 Land Cover Classification")
st.write("Upload a satellite image to classify it as **Cloudy**, **Desert**, **Green Area**, or **Water**.")

uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    resized_image = image.resize((255, 255))
    img_array = np.expand_dims(np.array(resized_image) / 255.0, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    # Display result
    st.success(f"✅ Predicted Class: **{predicted_class}**")
