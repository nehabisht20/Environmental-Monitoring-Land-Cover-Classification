import streamlit as st
import numpy as np
from PIL import Image
import os
import gdown
from tensorflow.keras.models import load_model

MODEL_ID = "1Y5bmVguQu7-RcIx0LGnKlhbtM5kN9EM9"
MODEL_PATH = "Modelenv.v1.h5"

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = download_and_load_model()

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

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🧠 Classify Image"):
        with st.spinner("Predicting..."):
            processed = preprocess_image(image)
            prediction = model.predict(processed)
            predicted_label = label_map[np.argmax(prediction)]

        st.success(f"✅ Predicted Class: **{predicted_label}**")
        st.bar_chart(prediction[0])
