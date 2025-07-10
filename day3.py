import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

# Load the trained model
@st.cache_resource
def load_trained_model():
    return load_model("Modelenv.v1.h5")

model = load_trained_model()
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

st.set_page_config(page_title="Land Cover Classification 🌍", layout="wide")

st.title("🌱 Environmental Monitoring")
st.subheader("🌤️ Classifying Satellite Images into Land Cover Types")

# Sidebar
st.sidebar.header("📤 Upload Satellite Image")
uploaded_file = st.sidebar.file_uploader("Choose a satellite image", type=['jpg', 'jpeg', 'png'])

st.sidebar.markdown("---")
st.sidebar.write("👈 Upload an image to see prediction")

# Display image and prediction
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
    st.info("Upload an image from the sidebar to begin.")

# Sample image section
with st.expander("🔍 View Sample Images for Each Class"):
    cols = st.columns(4)
    class_folders = {
        'Cloudy': 'dataset/Satellite Image data/cloudy',
        'Desert': 'dataset/Satellite Image data/desert',
        'Green_Area': 'dataset/Satellite Image data/green_area',
        'Water': 'dataset/Satellite Image data/water'
    }

    for i, (class_name, path) in enumerate(class_folders.items()):
        with cols[i]:
            st.markdown(f"**{class_name}**")
            try:
                image_files = [img for img in os.listdir(path) if img.endswith(('.png', '.jpg'))]
                for img_name in image_files[:2]:
                    img = Image.open(f"{path}/{img_name}")
                    st.image(img, use_column_width=True)
            except Exception as e:
                st.warning(f"Could not load images for {class_name}: {e}")
