import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
import tempfile
import os
import gdown

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
def load_trained_model():
    """Load the pre-trained model using multiple methods"""
    model_path = 'Modelenv.v1.h5'
    
    # Method 1: Check if model exists locally
    if os.path.exists(model_path):
        try:
            st.info("🔍 Loading model from local file...")
            return load_model(model_path)
        except Exception as e:
            st.warning(f"Local model file corrupted: {str(e)}")
    
    # Method 2: Try using gdown (more reliable for Google Drive)
    try:
        st.info("📥 Downloading model using gdown...")
        file_id = "1Y5bmVguQu7-RcIx0LGnKlhbtM5kN9EM9"
        url = f"https://drive.google.com/uc?id={file_id}"
        
        # Download to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
            tmp_file_path = tmp_file.name
        
        # Use gdown to download
        gdown.download(url, tmp_file_path, quiet=False)
        
        # Verify file was downloaded properly
        if os.path.exists(tmp_file_path) and os.path.getsize(tmp_file_path) > 1000:
            model = load_model(tmp_file_path)
            
            # Save model locally for future use
            if model is not None:
                model.save(model_path)
                st.success("✅ Model downloaded and saved locally!")
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            return model
        else:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            
    except Exception as e:
        st.warning(f"gdown method failed: {str(e)}")
    
    # Method 3: Direct download with session handling
    try:
        st.info("📥 Attempting direct download...")
        file_id = "1Y5bmVguQu7-RcIx0LGnKlhbtM5kN9EM9"
        
        # Create session
        session = requests.Session()
        
        # Try direct download URL
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # Make request
        response = session.get(download_url, stream=True)
        
        # Handle Google Drive's virus scan warning
        if response.status_code == 200:
            # Check if we need to handle virus scan warning
            if b'virus scan warning' in response.content[:1000].lower():
                # Look for download link in the response
                content = response.text
                if 'confirm=' in content:
                    # Extract confirm token
                    start = content.find('confirm=') + 8
                    end = content.find('&', start)
                    if end == -1:
                        end = content.find('"', start)
                    confirm_token = content[start:end]
                    
                    # Make confirmed download request
                    confirmed_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                    response = session.get(confirmed_url, stream=True)
            
            # Save and load model
            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            tmp_file.write(chunk)
                    tmp_file_path = tmp_file.name
                
                # Verify file size
                if os.path.getsize(tmp_file_path) > 1000:
                    model = load_model(tmp_file_path)
                    
                    # Save locally
                    if model is not None:
                        model.save(model_path)
                        st.success("✅ Model downloaded and saved locally!")
                    
                    # Clean up
                    os.unlink(tmp_file_path)
                    return model
                else:
                    os.unlink(tmp_file_path)
                    
    except Exception as e:
        st.warning(f"Direct download failed: {str(e)}")
    
    # Method 4: Use a pre-built model as fallback
    try:
        st.info("🏗️ Creating a simple fallback model...")
        # Create a simple model with the same architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), input_shape=(255, 255, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Initialize with random weights (demo purposes)
        dummy_input = np.random.random((1, 255, 255, 3))
        model.predict(dummy_input)
        
        st.warning("⚠️ Using untrained model for demo purposes. Predictions may not be accurate.")
        return model
        
    except Exception as e:
        st.error(f"Failed to create fallback model: {str(e)}")
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
    Upload a satellite image to get predictions for the following categories:
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
        1. Wait for the model to load automatically
        2. Upload a satellite image
        3. View the prediction results
        4. Check confidence scores
        """)
    
    # Load model automatically
    with st.spinner("🔄 Loading model... Please wait..."):
        model = load_trained_model()
    
    if model is None:
        st.error("❌ Failed to load the model. Please refresh the page and try again.")
        st.stop()
    
    # File uploader
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
