# Face Swapping Implementation

Production-ready face swapping application built with Python, OpenCV, and Dlib. This project demonstrates advanced computer vision techniques for realistic face swapping with a clean, user-friendly interface.

## Features

- **Real-time Face Swapping**: Upload images and swap faces instantly
- **Synthetic Demo Mode**: Try the application with AI-generated faces
- **Configurable Parameters**: Adjust processing settings for optimal results
- **Web Interface**: Beautiful Streamlit-based UI for easy interaction
- **Modern Architecture**: Type hints, error handling, and clean code practices
- **Extensible Design**: Easy to integrate with modern ML libraries

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kryptologyst/Face-Swapping-Implementation.git
   cd Face-Swapping-Implementation
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Dlib model**
   ```bash
   # Create models directory
   mkdir -p models
   
   # Download the shape predictor (you'll need to download this manually)
   # Place shape_predictor_68_face_landmarks.dat in the models/ directory
   ```

### Running the Application

#### Web Interface (Recommended)
```bash
streamlit run web_app/app.py
```

#### Command Line Interface
```bash
python src/face_swapper.py
```

## 📁 Project Structure

```
face-swapping-implementation/
├── src/                    # Source code
│   ├── face_swapper.py     # Core face swapping logic
│   ├── config_manager.py   # Configuration management
│   └── data_generator.py   # Synthetic data generation
├── web_app/               # Web interface
│   └── app.py             # Streamlit application
├── data/                  # Data directory
│   └── synthetic/         # Generated synthetic faces
├── models/                # Model files
│   └── shape_predictor_68_face_landmarks.dat
├── config/                # Configuration files
├── tests/                 # Test files
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🛠️ Technical Details

### Core Technologies

- **OpenCV**: Image processing and computer vision
- **Dlib**: Facial landmark detection
- **Streamlit**: Web interface framework
- **NumPy**: Numerical computations
- **PIL/Pillow**: Image manipulation

### Face Swapping Pipeline

1. **Face Detection**: Detect facial landmarks using Dlib's 68-point model
2. **Triangulation**: Create Delaunay triangulation mesh based on landmarks
3. **Warping**: Apply affine transformations to map source face to target
4. **Blending**: Use seamless cloning for natural-looking results

### Key Improvements Over Original

- ✅ **Type Hints**: Full type annotation for better code clarity
- ✅ **Error Handling**: Comprehensive error handling and logging
- ✅ **Configuration Management**: YAML/JSON-based configuration system
- ✅ **Web Interface**: Modern Streamlit-based UI
- ✅ **Synthetic Data**: AI-generated test data for demonstration
- ✅ **Modular Design**: Clean separation of concerns
- ✅ **Documentation**: Comprehensive docstrings and comments

## Usage Examples

### Basic Face Swapping

```python
from src.face_swapper import FaceSwapper, FaceSwapConfig

# Create configuration
config = FaceSwapConfig(
    resize_width=500,
    clone_method=cv2.NORMAL_CLONE
)

# Initialize face swapper
swapper = FaceSwapper(config)

# Perform face swap
result = swapper.swap_faces("source.jpg", "target.jpg")

# Save result
if result is not None:
    swapper.save_result(result, "output.jpg")
```

### Configuration Management

```python
from src.config_manager import config_manager

# Load configuration
config = config_manager.get_config()

# Update settings
config_manager.update_config(
    resize_width=600,
    clone_method=cv2.MIXED_CLONE
)

# Save configuration
config_manager.save_config()
```

### Synthetic Data Generation

```python
from src.data_generator import SyntheticFaceGenerator

# Create generator
generator = SyntheticFaceGenerator("data/synthetic")

# Generate dataset
faces = generator.generate_dataset(num_faces=10)
```

## Configuration

The application supports configuration through YAML or JSON files. Create a `config/config.yaml` file:

```yaml
model:
  predictor_path: "models/shape_predictor_68_face_landmarks.dat"
  confidence_threshold: 0.5
  max_faces: 1

processing:
  resize_width: 500
  clone_method: 1
  border_mode: 4
  quality: 95

ui:
  theme: "light"
  show_landmarks: false
  show_triangles: false
  auto_save: true

log_level: "INFO"
output_dir: "output"
temp_dir: "temp"
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_face_swapper.py
```

## 🔧 Development

### Code Quality

The project follows modern Python best practices:

- **Type Hints**: All functions have type annotations
- **PEP 8**: Code follows Python style guidelines
- **Docstrings**: Google-style docstrings for all functions
- **Error Handling**: Comprehensive exception handling
- **Logging**: Structured logging throughout

### Code Formatting

```bash
# Format code with Black
black src/ web_app/ tests/

# Check code style with Flake8
flake8 src/ web_app/ tests/

# Type checking with MyPy
mypy src/
```

## Performance Considerations

- **Image Size**: Larger images take longer to process but may yield better results
- **Face Detection**: Clear, well-lit images with forward-facing faces work best
- **Memory Usage**: Processing large images may require significant RAM
- **GPU Acceleration**: Consider using OpenCV with CUDA support for faster processing

## ⚠️ Ethical Considerations

This application is intended for:

- ✅ **Educational purposes**: Learning computer vision techniques
- ✅ **Entertainment**: Fun applications and filters
- ✅ **Research**: Academic and research applications

**Please use responsibly:**

- ❌ Do not use without consent
- ❌ Do not create misleading content
- ❌ Respect privacy and ethical guidelines
- ❌ Follow local laws and regulations

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Dlib**: For facial landmark detection
- **OpenCV**: For computer vision capabilities
- **Streamlit**: For the web interface framework
- **Original Implementation**: Based on the original face swapping concept

## Future Enhancements

- [ ] **GPU Acceleration**: CUDA support for faster processing
- [ ] **Video Support**: Face swapping in video files
- [ ] **Multiple Faces**: Support for multiple faces in one image
- [ ] **Real-time Processing**: Live camera face swapping
- [ ] **Advanced Models**: Integration with modern face detection models
- [ ] **Batch Processing**: Process multiple images at once
- [ ] **API Endpoints**: REST API for integration with other applications


# Face-Swapping-Implementation
