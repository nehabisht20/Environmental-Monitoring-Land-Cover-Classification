import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load the model once
@st.cache_resource
def load_my_model():
    return load_model("model/Modelenv.v1.h5")

model = load_my_model()

labels = {
    r"/content/dataset/Satellite Image data/cloudy": "Cloudy",
    r"/content/dataset/Satellite Image data/desert": "Desert",
    r"/content/dataset/Satellite Image data/green_area": "Green_Area",
    r"/content/dataset/Satellite Image data/water": "Water",
}

label_map = list(labels.values())

def preprocess_image(image: Image.Image):
    image = image.resize((256, 256))
    image = np.array(image)
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    image = image / 255.0
    return np.expand_dims(image, axis=0)

st.set_page_config(page_title="Land Cover Classifier 🌍", layout="wide")
st.title("🛰️ Satellite Land Cover Classification")
st.markdown("Upload a satellite image and let the AI classify it into one of the land cover types.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🧠 Classify Image"):
        with st.spinner("Predicting..."):
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)
            predicted_label = label_map[np.argmax(prediction)]

        st.success(f"✅ Predicted Class: **{predicted_label}**")
        st.bar_chart(prediction[0])
