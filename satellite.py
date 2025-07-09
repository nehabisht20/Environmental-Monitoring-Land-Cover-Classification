import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Environmental Monitoring & Land Cover Classification",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4682B4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
    }
    .prediction-card {
        background-color: #e6f3ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #4682B4;
        margin: 1rem 0;
    }
    .model-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4682B4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Class names
CLASS_NAMES = ['Cloudy', 'Desert', 'Green_Area', 'Water']
CLASS_COLORS = ['#708090', '#F4A460', '#228B22', '#1E90FF']

# Helper functions
@st.cache_resource
def create_model():
    """Create and return a pre-configured model architecture"""
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(255, 255, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_image(img, target_size=(255, 255)):
    """Preprocess image for prediction"""
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_image_demo(img_array):
    """Make a demo prediction (random for demonstration)"""
    # Generate random probabilities for demo
    probabilities = np.random.dirichlet(np.ones(4), size=1)[0]
    predicted_class = np.argmax(probabilities)
    confidence = np.max(probabilities)
    return predicted_class, confidence, probabilities

def create_prediction_chart(probabilities, class_names):
    """Create a bar chart for class probabilities"""
    fig = px.bar(
        x=class_names,
        y=probabilities,
        labels={'x': 'Land Cover Class', 'y': 'Probability'},
        title="Class Probabilities",
        color=probabilities,
        color_continuous_scale='viridis'
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title="Land Cover Class",
        yaxis_title="Probability",
        height=400
    )
    return fig

def display_sample_images():
    """Display sample images for each class"""
    st.markdown('<div class="sub-header">Sample Images by Class</div>', unsafe_allow_html=True)
    
    # Create sample data
    sample_data = {
        'Cloudy': '☁️ Cloudy satellite images show areas covered by clouds',
        'Desert': '🏜️ Desert images show arid, sandy landscapes',
        'Green_Area': '🌳 Green areas show vegetation and forests',
        'Water': '🌊 Water bodies like lakes, rivers, and oceans'
    }
    
    cols = st.columns(4)
    for i, (class_name, description) in enumerate(sample_data.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: {CLASS_COLORS[i]};">{class_name}</h3>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)

# Main application
def main():
    # Header
    st.markdown('<div class="main-header">🛰️ Environmental Monitoring & Land Cover Classification</div>', unsafe_allow_html=True)
    
    # Initialize model in session state
    if not st.session_state.model_loaded:
        with st.spinner("Initializing model..."):
            st.session_state.model = create_model()
            st.session_state.model_loaded = True
    
    # Sidebar with model info
    st.sidebar.title("🔧 Model Information")
    st.sidebar.markdown("""
    <div class="model-info">
        <h4>✅ Model Ready</h4>
        <p><strong>Architecture:</strong> CNN</p>
        <p><strong>Input Size:</strong> 255x255 RGB</p>
        <p><strong>Classes:</strong> 4 land cover types</p>
        <p><strong>Status:</strong> Ready for prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Image Prediction", "📊 Batch Analysis", "📈 Model Performance", "ℹ️ About"])
    
    with tab1:
        st.markdown('<div class="sub-header">Upload Satellite Image for Classification</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a satellite image",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a satellite image for land cover classification (JPG, JPEG, or PNG format)"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Image", use_column_width=True)
                
                # Image details
                st.info(f"📄 **File:** {uploaded_file.name}")
                st.info(f"📐 **Size:** {img.size[0]} x {img.size[1]} pixels")
                st.info(f"🎨 **Mode:** {img.mode}")
                
                # Prediction button
                if st.button("🔍 Classify Image", type="primary"):
                    with st.spinner("Analyzing image..."):
                        # Preprocess and predict
                        img_array = preprocess_image(img)
                        predicted_class, confidence, probabilities = predict_image_demo(img_array)
                        
                        # Store results in session state
                        st.session_state.prediction_results = {
                            'predicted_class': predicted_class,
                            'confidence': confidence,
                            'probabilities': probabilities,
                            'class_name': CLASS_NAMES[predicted_class]
                        }
        
        with col2:
            if hasattr(st.session_state, 'prediction_results'):
                results = st.session_state.prediction_results
                
                # Display prediction result
                st.markdown(f"""
                <div class="prediction-card">
                    <h2 style="color: {CLASS_COLORS[results['predicted_class']]};">
                        Predicted Class: {results['class_name']}
                    </h2>
                    <h3>Confidence: {results['confidence']:.2%}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Display probability chart
                fig = create_prediction_chart(results['probabilities'], CLASS_NAMES)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed probabilities
                st.subheader("Detailed Probabilities")
                prob_df = pd.DataFrame({
                    'Class': CLASS_NAMES,
                    'Probability': results['probabilities'],
                    'Percentage': [f"{p:.2%}" for p in results['probabilities']]
                })
                st.dataframe(prob_df, use_container_width=True)
    
    with tab2:
        st.markdown('<div class="sub-header">Batch Image Analysis</div>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload multiple satellite images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload multiple images for batch classification (JPG, JPEG, or PNG format)"
        )
        
        if uploaded_files:
            st.info(f"📁 **{len(uploaded_files)} images** uploaded for batch processing")
            
            if st.button("🔍 Analyze All Images", type="primary"):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    img = Image.open(uploaded_file)
                    img_array = preprocess_image(img)
                    predicted_class, confidence, probabilities = predict_image_demo(img_array)
                    
                    results.append({
                        'filename': uploaded_file.name,
                        'predicted_class': CLASS_NAMES[predicted_class],
                        'confidence': confidence,
                        'probabilities': probabilities
                    })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("Analysis complete!")
                
                # Display results
                st.subheader("Batch Analysis Results")
                
                # Create summary dataframe
                summary_df = pd.DataFrame([{
                    'Filename': r['filename'],
                    'Predicted Class': r['predicted_class'],
                    'Confidence': f"{r['confidence']:.2%}"
                } for r in results])
                
                st.dataframe(summary_df, use_container_width=True)
                
                # Class distribution chart
                class_counts = pd.Series([r['predicted_class'] for r in results]).value_counts()
                fig = px.pie(
                    values=class_counts.values,
                    names=class_counts.index,
                    title="Distribution of Predicted Classes",
                    color_discrete_sequence=CLASS_COLORS
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Images", len(results))
                with col2:
                    avg_confidence = np.mean([r['confidence'] for r in results])
                    st.metric("Average Confidence", f"{avg_confidence:.2%}")
                with col3:
                    most_common = class_counts.index[0]
                    st.metric("Most Common Class", most_common)
                
                # Download results
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="batch_analysis_results.csv",
                    mime="text/csv"
                )
    
    with tab3:
        st.markdown('<div class="sub-header">Model Performance & Architecture</div>', unsafe_allow_html=True)
        
        # Model architecture info
        st.subheader("Model Architecture")
        if st.session_state.model:
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Total Parameters:** {st.session_state.model.count_params():,}")
                st.info(f"**Input Shape:** {st.session_state.model.input_shape}")
                st.info(f"**Output Classes:** {len(CLASS_NAMES)}")
            
            with col2:
                st.info("**Model Type:** Convolutional Neural Network")
                st.info("**Framework:** TensorFlow/Keras")
                st.info("**Image Size:** 255x255 pixels")
        
        # Model layers visualization
        st.subheader("Model Layers")
        if st.session_state.model:
            layers_info = []
            for i, layer in enumerate(st.session_state.model.layers):
                # Safely get output shape
                try:
                    if hasattr(layer, 'output_shape'):
                        output_shape = str(layer.output_shape)
                    else:
                        output_shape = "N/A"
                except:
                    output_shape = "N/A"
                
                layers_info.append({
                    'Layer': i + 1,
                    'Type': layer.__class__.__name__,
                    'Output Shape': output_shape,
                    'Parameters': layer.count_params()
                })
            
            layers_df = pd.DataFrame(layers_info)
            st.dataframe(layers_df, use_container_width=True)
        
        # Demo training curves
        st.subheader("Sample Training Curves")
        st.info("💡 These are sample training curves for demonstration purposes")
        
        # Generate sample training data
        epochs = np.arange(1, 26)
        train_acc = 0.3 + 0.6 * (1 - np.exp(-epochs/8)) + np.random.normal(0, 0.02, 25)
        val_acc = 0.3 + 0.5 * (1 - np.exp(-epochs/10)) + np.random.normal(0, 0.03, 25)
        train_loss = 1.5 * np.exp(-epochs/5) + 0.1 + np.random.normal(0, 0.05, 25)
        val_loss = 1.6 * np.exp(-epochs/6) + 0.15 + np.random.normal(0, 0.08, 25)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Training & Validation Accuracy', 'Training & Validation Loss')
        )
        
        # Accuracy plot
        fig.add_trace(
            go.Scatter(x=epochs, y=train_acc, name='Training Accuracy', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_acc, name='Validation Accuracy', line=dict(color='red')),
            row=1, col=1
        )
        
        # Loss plot
        fig.add_trace(
            go.Scatter(x=epochs, y=train_loss, name='Training Loss', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_loss, name='Validation Loss', line=dict(color='red')),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Epochs")
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown('<div class="sub-header">About This Application</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🛰️ Environmental Monitoring & Land Cover Classification
        
        This application uses deep learning to classify satellite images into different land cover types:
        
        - **🌥️ Cloudy**: Areas covered by clouds
        - **🏜️ Desert**: Arid and sandy landscapes
        - **🌳 Green Area**: Vegetation and forested regions
        - **🌊 Water**: Water bodies (lakes, rivers, oceans)
        
        ### 🔧 How It Works
        
        1. **Ready to Use**: The model is pre-loaded and ready for immediate use
        2. **Image Upload**: Upload satellite images in JPG, JPEG, or PNG format
        3. **Instant Classification**: Get predictions with confidence scores
        4. **Batch Processing**: Analyze multiple images at once
        
        ### 📊 Technical Details
        
        - **Model Architecture**: Convolutional Neural Network (CNN)
        - **Input Size**: 255x255 RGB images
        - **Classes**: 4 land cover types
        - **Framework**: TensorFlow/Keras
        - **Preprocessing**: Automatic image resizing and normalization
        - **Supported Formats**: JPG, JPEG, PNG
        
        ### 🚀 Getting Started
        
        1. Navigate to the "Single Image Prediction" tab
        2. Upload a satellite image (JPG, JPEG, or PNG)
        3. Click "Classify Image" to get results
        4. View detailed probabilities and confidence scores
        
        ### 💡 Tips for Best Results
        
        - Use high-quality satellite images
        - Ensure images are clear and well-lit
        - Images should show distinct land cover features
        - Optimal image size: 255x255 pixels or larger
        - Supported formats: JPG, JPEG, PNG only
        
        ### 🔬 Demo Mode
        
        **Note**: This application is currently running in demo mode with simulated predictions. 
        In a production environment, you would load your actual trained model weights.
        """)
        
        # Display sample images section
        display_sample_images()
        
        # Usage statistics
        st.subheader("Usage Guidelines")
        
        guidelines = {
            "✅ Supported": [
                "JPG, JPEG, PNG image formats",
                "Satellite and aerial imagery",
                "RGB color images",
                "Single or batch processing"
            ],
            "❌ Not Supported": [
                "GIF, BMP, TIFF formats",
                "Grayscale images",
                "Very low resolution images",
                "Non-satellite imagery"
            ]
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ Supported Features:**")
            for item in guidelines["✅ Supported"]:
                st.markdown(f"- {item}")
        
        with col2:
            st.markdown("**❌ Not Supported:**")
            for item in guidelines["❌ Not Supported"]:
                st.markdown(f"- {item}")

if __name__ == "__main__":
    main()
