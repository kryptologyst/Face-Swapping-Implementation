"""
Streamlit web interface for face swapping application.
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
import logging
from typing import Optional, Tuple
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from face_swapper import FaceSwapper, FaceSwapConfig
from config_manager import config_manager
from data_generator import SyntheticFaceGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Face Swapping App",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


def load_image(file) -> Optional[np.ndarray]:
    """
    Load image from uploaded file.
    
    Args:
        file: Uploaded file object.
        
    Returns:
        Image as numpy array or None if loading fails.
    """
    try:
        # Convert uploaded file to bytes
        bytes_data = file.getvalue()
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(bytes_data, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("Failed to decode image. Please upload a valid image file.")
            return None
        
        return image
        
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None


def display_image(image: np.ndarray, caption: str, use_column_width: bool = True) -> None:
    """
    Display image in Streamlit.
    
    Args:
        image: Image array.
        caption: Caption for the image.
        use_column_width: Whether to use column width.
    """
    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption=caption, use_column_width=use_column_width)


def create_synthetic_faces() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Create synthetic faces for demonstration.
    
    Returns:
        Tuple of (source_face, target_face) or (None, None) if creation fails.
    """
    try:
        generator = SyntheticFaceGenerator()
        
        # Generate two different faces
        source_face = generator.generate_synthetic_face(1)
        target_face = generator.generate_synthetic_face(2)
        
        return source_face, target_face
        
    except Exception as e:
        st.error(f"Error creating synthetic faces: {e}")
        return None, None


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔄 Face Swapping Application</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Configuration options
    resize_width = st.sidebar.slider("Resize Width", 300, 800, 500)
    clone_method = st.sidebar.selectbox(
        "Clone Method",
        ["Normal Clone", "Mixed Clone", "Monochrome Transfer"],
        index=0
    )
    
    clone_method_map = {
        "Normal Clone": cv2.NORMAL_CLONE,
        "Mixed Clone": cv2.MIXED_CLONE,
        "Monochrome Transfer": cv2.MONOCHROME_TRANSFER
    }
    
    show_landmarks = st.sidebar.checkbox("Show Landmarks", False)
    show_triangles = st.sidebar.checkbox("Show Triangles", False)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🖼️ Upload Images", "🎨 Synthetic Demo", "ℹ️ About"])
    
    with tab1:
        st.header("Upload Your Images")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Source Face")
            source_file = st.file_uploader(
                "Choose source face image",
                type=['jpg', 'jpeg', 'png'],
                key="source"
            )
            
            if source_file is not None:
                source_image = load_image(source_file)
                if source_image is not None:
                    display_image(source_image, "Source Face")
        
        with col2:
            st.subheader("Target Face")
            target_file = st.file_uploader(
                "Choose target face image",
                type=['jpg', 'jpeg', 'png'],
                key="target"
            )
            
            if target_file is not None:
                target_image = load_image(target_file)
                if target_image is not None:
                    display_image(target_image, "Target Face")
        
        # Face swapping
        if source_file is not None and target_file is not None:
            if st.button("🔄 Swap Faces", type="primary"):
                with st.spinner("Processing face swap..."):
                    try:
                        # Create configuration
                        config = FaceSwapConfig(
                            resize_width=resize_width,
                            clone_method=clone_method_map[clone_method]
                        )
                        
                        # Initialize face swapper
                        swapper = FaceSwapper(config)
                        
                        # Save uploaded files temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_source:
                            tmp_source.write(source_file.getvalue())
                            tmp_source_path = tmp_source.name
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_target:
                            tmp_target.write(target_file.getvalue())
                            tmp_target_path = tmp_target.name
                        
                        # Perform face swap
                        result = swapper.swap_faces(tmp_source_path, tmp_target_path)
                        
                        # Clean up temporary files
                        os.unlink(tmp_source_path)
                        os.unlink(tmp_target_path)
                        
                        if result is not None:
                            st.success("Face swap completed successfully!")
                            
                            # Display result
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                display_image(source_image, "Source")
                            
                            with col2:
                                display_image(target_image, "Target")
                            
                            with col3:
                                display_image(result, "Result")
                            
                            # Download button
                            result_bytes = cv2.imencode('.jpg', result)[1].tobytes()
                            st.download_button(
                                label="📥 Download Result",
                                data=result_bytes,
                                file_name="face_swap_result.jpg",
                                mime="image/jpeg"
                            )
                        else:
                            st.error("Face swap failed. Please try with different images.")
                            
                    except Exception as e:
                        st.error(f"Error during face swap: {e}")
                        logger.error(f"Face swap error: {e}")
    
    with tab2:
        st.header("Synthetic Face Demo")
        st.markdown("""
        <div class="info-box">
            <strong>Demo Mode:</strong> This tab demonstrates face swapping using synthetic faces 
            generated by the application. No real images are used.
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎲 Generate Synthetic Faces", type="primary"):
            with st.spinner("Generating synthetic faces..."):
                try:
                    source_face, target_face = create_synthetic_faces()
                    
                    if source_face is not None and target_face is not None:
                        st.success("Synthetic faces generated successfully!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            display_image(source_face, "Synthetic Source Face")
                        
                        with col2:
                            display_image(target_face, "Synthetic Target Face")
                        
                        # Perform face swap on synthetic faces
                        if st.button("🔄 Swap Synthetic Faces"):
                            with st.spinner("Processing synthetic face swap..."):
                                try:
                                    # Save synthetic faces temporarily
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_source:
                                        cv2.imwrite(tmp_source.name, source_face)
                                        tmp_source_path = tmp_source.name
                                    
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_target:
                                        cv2.imwrite(tmp_target.name, target_face)
                                        tmp_target_path = tmp_target.name
                                    
                                    # Create configuration
                                    config = FaceSwapConfig(
                                        resize_width=resize_width,
                                        clone_method=clone_method_map[clone_method]
                                    )
                                    
                                    # Initialize face swapper
                                    swapper = FaceSwapper(config)
                                    
                                    # Perform face swap
                                    result = swapper.swap_faces(tmp_source_path, tmp_target_path)
                                    
                                    # Clean up temporary files
                                    os.unlink(tmp_source_path)
                                    os.unlink(tmp_target_path)
                                    
                                    if result is not None:
                                        st.success("Synthetic face swap completed!")
                                        
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            display_image(source_face, "Source")
                                        
                                        with col2:
                                            display_image(target_face, "Target")
                                        
                                        with col3:
                                            display_image(result, "Result")
                                    else:
                                        st.error("Synthetic face swap failed.")
                                        
                                except Exception as e:
                                    st.error(f"Error during synthetic face swap: {e}")
                                    logger.error(f"Synthetic face swap error: {e}")
                    else:
                        st.error("Failed to generate synthetic faces.")
                        
                except Exception as e:
                    st.error(f"Error generating synthetic faces: {e}")
                    logger.error(f"Synthetic face generation error: {e}")
    
    with tab3:
        st.header("About This Application")
        
        st.markdown("""
        ## 🔄 Face Swapping Application
        
        This application demonstrates modern face swapping techniques using computer vision and machine learning.
        
        ### ✨ Features
        
        - **Real-time Face Swapping**: Upload your own images and swap faces instantly
        - **Synthetic Demo**: Try the application with generated synthetic faces
        - **Configurable Parameters**: Adjust processing settings for optimal results
        - **Modern Architecture**: Built with type hints, error handling, and clean code practices
        
        ### 🛠️ Technical Details
        
        - **Face Detection**: Uses Dlib's facial landmark detection
        - **Image Processing**: OpenCV for image manipulation and blending
        - **Triangulation**: Delaunay triangulation for smooth face mapping
        - **Seamless Cloning**: Advanced blending techniques for realistic results
        
        ### 📚 How It Works
        
        1. **Face Detection**: Detect facial landmarks in both source and target images
        2. **Triangulation**: Create triangular mesh based on facial landmarks
        3. **Warping**: Apply affine transformations to map source face to target
        4. **Blending**: Use seamless cloning for natural-looking results
        
        ### ⚠️ Ethical Considerations
        
        This application is intended for educational and entertainment purposes only. 
        Please use responsibly and respect others' privacy and consent.
        
        ### 🔧 Requirements
        
        - Python 3.8+
        - OpenCV
        - Dlib
        - Streamlit
        - NumPy
        
        ### 📖 Usage Instructions
        
        1. **Upload Images**: Use the "Upload Images" tab to upload your source and target face images
        2. **Configure Settings**: Adjust parameters in the sidebar for optimal results
        3. **Process**: Click "Swap Faces" to perform the face swap
        4. **Download**: Save your result using the download button
        
        For best results, use clear, well-lit images with faces facing forward.
        """)


if __name__ == "__main__":
    main()
