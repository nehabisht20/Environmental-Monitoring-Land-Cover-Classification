import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="Environmental Monitoring & Land Cover Classification",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8f0;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 20px 0;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        height: 20px;
        border-radius: 10px;
        overflow: hidden;
        margin: 5px 0;
    }
    .confidence-fill {
        height: 100%;
        background-color: #2E8B57;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_from_upload(uploaded_model_file):
    """Load model from uploaded file"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
            tmp_file.write(uploaded_model_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load the model
        model = load_model(tmp_file_path)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return model
    except Exception as e:
        st.error(f"Error loading uploaded model: {str(e)}")
        return None

@st.cache_resource
def load_local_model():
    """Load model from local file if it exists"""
    if os.path.exists('Modelenv.v1.h5'):
        try:
            return load_model('Modelenv.v1.h5')
        except Exception as e:
            st.error(f"Error loading local model: {str(e)}")
            return None
    return None

def preprocess_image(uploaded_image):
    """Preprocess the uploaded image for prediction"""
    # Convert to RGB if necessary
    if uploaded_image.mode != 'RGB':
        uploaded_image = uploaded_image.convert('RGB')
    
    # Resize to match model input size
    img_resized = uploaded_image.resize((255, 255))
    
    # Convert to array and normalize
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array

def predict_land_cover(model, processed_image):
    """Make prediction using the loaded model"""
    predictions = model.predict(processed_image)
    return predictions[0]

def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 Environmental Monitoring & Land Cover Classification</h1>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    This application uses a Convolutional Neural Network (CNN) to classify satellite images into different land cover types.
    Upload your trained model and satellite images to get predictions for the following categories:
    - **🌥️ Cloudy**: Cloud-covered areas
    - **🏜️ Desert**: Desert and arid regions
    - **🌿 Green Area**: Vegetation and forested areas
    - **💧 Water**: Water bodies (rivers, lakes, oceans)
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.info("""
        **Model Architecture:**
        - CNN with 3 Convolutional layers
        - Input size: 255x255 pixels
        - 4 output classes
        - Trained on satellite imagery dataset
        """)
        
        st.header("🚀 How to Use")
        st.markdown("""
        1. Upload your trained model file (.h5)
        2. Upload a satellite image
        3. Wait for the model to process
        4. View the prediction results
        """)
        
        st.header("📁 Download Model")
        st.markdown("""
        Download your model from Google Drive:
        [Modelenv.v1.h5](https://drive.google.com/file/d/1Y5bmVguQu7-RcIx0LGnKlhbtM5kN9EM9/view)
        """)
    
    # Try to load local model first
    model = load_local_model()
    
    if model is None:
        st.info("🔍 No local model found. Please upload your model file.")
        
        # Model upload section
        st.subheader("📤 Upload Model")
        uploaded_model = st.file_uploader(
            "Upload your trained model file",
            type=['h5'],
            help="Upload the Modelenv.v1.h5 file that you downloaded from Google Drive"
        )
        
        if uploaded_model is not None:
            with st.spinner("Loading model..."):
                model = load_model_from_upload(uploaded_model)
            
            if model is not None:
                st.success("✅ Model loaded successfully!")
            else:
                st.error("❌ Failed to load the model. Please check the file format.")
                return
        else:
            st.warning("⚠️ Please upload your model file to continue.")
            return
    else:
        st.success("✅ Model loaded from local file!")
    
    # Image upload and prediction section
    st.subheader("📸 Upload Satellite Image")
    
    # File uploader for images
    uploaded_file = st.file_uploader(
        "Choose a satellite image...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload a satellite image for land cover classification"
    )
    
    if uploaded_file is not None:
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Uploaded Image")
            # Display uploaded image
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption="Original Image", use_column_width=True)
            
            # Image info
            st.write(f"**Image Size:** {uploaded_image.size}")
            st.write(f"**Image Mode:** {uploaded_image.mode}")
        
        with col2:
            st.subheader("🔍 Prediction Results")
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                processed_image = preprocess_image(uploaded_image)
                predictions = predict_land_cover(model, processed_image)
            
            # Define class names and emojis
            class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']
            class_emojis = ['🌥️', '🏜️', '🌿', '💧']
            class_colors = ['#87CEEB', '#DEB887', '#228B22', '#4682B4']
            
            # Get predicted class
            predicted_class_idx = np.argmax(predictions)
            predicted_class = class_names[predicted_class_idx]
            confidence = predictions[predicted_class_idx] * 100
            
            # Display main prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h3>{class_emojis[predicted_class_idx]} Predicted Class: {predicted_class}</h3>
                <h4>Confidence: {confidence:.2f}%</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Display all predictions with confidence bars
            st.subheader("📊 All Predictions")
            
            # Sort predictions by confidence
            sorted_indices = np.argsort(predictions)[::-1]
            
            for i, idx in enumerate(sorted_indices):
                class_name = class_names[idx]
                emoji = class_emojis[idx]
                conf = predictions[idx] * 100
                color = class_colors[idx]
                
                # Create confidence bar
                st.markdown(f"""
                <div style="margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: bold;">{emoji} {class_name}</span>
                        <span style="font-weight: bold;">{conf:.2f}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {conf}%; background-color: {color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Additional analysis section
        st.subheader("📈 Detailed Analysis")
        
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.write("**Prediction Summary:**")
            summary_data = {
                'Class': [f"{class_emojis[i]} {class_names[i]}" for i in range(len(class_names))],
                'Confidence (%)': [f"{predictions[i]*100:.2f}%" for i in range(len(class_names))],
                'Rank': [f"#{np.where(sorted_indices == i)[0][0] + 1}" for i in range(len(class_names))]
            }
            st.dataframe(summary_data, use_container_width=True)
        
        with col4:
            st.write("**Interpretation Guide:**")
            st.markdown("""
            - **High Confidence (>80%)**: Very reliable prediction
            - **Medium Confidence (50-80%)**: Good prediction, but verify
            - **Low Confidence (<50%)**: Uncertain, manual verification needed
            """)
            
            # Provide interpretation based on confidence
            if confidence > 80:
                st.success("🎯 High confidence prediction - Very reliable!")
            elif confidence > 50:
                st.warning("⚠️ Medium confidence - Good prediction, but consider verification")
            else:
                st.error("❌ Low confidence - Manual verification recommended")
    
    else:
        st.info("👆 Please upload a satellite image to get started!")
        
        # Show example section
        st.subheader("🖼️ Example Images")
        st.markdown("""
        Try uploading satellite images containing:
        - Cloud formations over land or water
        - Desert landscapes with sand dunes
        - Forest areas with dense vegetation
        - Rivers, lakes, or coastal areas
        """)

if __name__ == "__main__":
    main()
