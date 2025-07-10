import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import gdown

# Download model from Google Drive
MODEL_URL = "https://drive.google.com/uc?export=download&id=1Y5bmVguQu7-RcIx0LGnKlhbtM5kN9EM9"
MODEL_PATH = "Modelenv.v1.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the model
model = load_model(MODEL_PATH)
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# Streamlit UI
st.set_page_config(page_title="Land Cover Classification 🌍", layout="wide")
st.title("🌱 Environmental Monitoring")
st.subheader("🌤️ Classifying Satellite Images into Land Cover Types")

# Sidebar: Upload Image
st.sidebar.header("📤 Upload Satellite Image")
uploaded_file = st.sidebar.file_uploader("Choose a satellite image", type=['jpg', 'jpeg', 'png'])

# Display uploaded image and prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    image_resized = image.resize((255, 255))
    img_array = img_to_array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]

    st.markdown(f"### 🧠 Prediction: **{predicted_class}**")
    
    # Plot probabilities
    st.markdown("#### 📊 Class Probabilities")
    fig, ax = plt.subplots()
    ax.bar(class_names, prediction, color='skyblue')
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    st.pyplot(fig)
else:
    st.info("Upload an image from the sidebar to see prediction.")
