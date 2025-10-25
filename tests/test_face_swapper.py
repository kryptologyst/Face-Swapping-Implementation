"""
Test suite for the face swapping application.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from face_swapper import FaceSwapper, FaceSwapConfig, FaceDetector
from config_manager import ConfigManager, AppConfig
from data_generator import SyntheticFaceGenerator, FaceLandmarkGenerator


class TestFaceDetector:
    """Test cases for FaceDetector class."""
    
    def test_face_detector_initialization(self):
        """Test face detector initialization."""
        # This test would require the actual model file
        # For now, we'll test the error handling
        with pytest.raises(FileNotFoundError):
            FaceDetector("nonexistent_model.dat")
    
    def test_get_landmarks_no_face(self):
        """Test landmark extraction with no face."""
        # Create a blank image
        blank_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # This would require the actual model file to test properly
        # For now, we'll just test that the method exists
        assert hasattr(FaceDetector, 'get_landmarks')


class TestFaceSwapper:
    """Test cases for FaceSwapper class."""
    
    def test_face_swapper_initialization(self):
        """Test face swapper initialization."""
        config = FaceSwapConfig()
        # This would require the actual model file
        # For now, we'll test the error handling
        with pytest.raises(FileNotFoundError):
            FaceSwapper(config)
    
    def test_load_image_nonexistent(self):
        """Test loading nonexistent image."""
        config = FaceSwapConfig()
        # Mock the detector to avoid model file requirement
        swapper = FaceSwapper.__new__(FaceSwapper)
        swapper.config = config
        
        result = swapper.load_image("nonexistent.jpg")
        assert result is None
    
    def test_save_result(self):
        """Test saving result image."""
        config = FaceSwapConfig()
        swapper = FaceSwapper.__new__(FaceSwapper)
        swapper.config = config
        
        # Create a test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            result = swapper.save_result(test_image, tmp_path)
            assert result is True
            assert Path(tmp_path).exists()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestConfigManager:
    """Test cases for ConfigManager class."""
    
    def test_config_manager_initialization(self):
        """Test config manager initialization."""
        config_manager = ConfigManager()
        assert isinstance(config_manager.config, AppConfig)
    
    def test_load_default_config(self):
        """Test loading default configuration."""
        config_manager = ConfigManager()
        config = config_manager._load_default_config()
        
        assert config.model.predictor_path == "models/shape_predictor_68_face_landmarks.dat"
        assert config.processing.resize_width == 500
        assert config.ui.theme == "light"
    
    def test_update_config(self):
        """Test configuration updates."""
        config_manager = ConfigManager()
        
        # Update configuration
        config_manager.update_config(log_level="DEBUG")
        assert config_manager.config.log_level == "DEBUG"
        
        # Test unknown parameter
        config_manager.update_config(unknown_param="test")
        # Should not raise an error, just log a warning


class TestSyntheticFaceGenerator:
    """Test cases for SyntheticFaceGenerator class."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = SyntheticFaceGenerator()
        assert generator.output_dir == Path("data/synthetic")
    
    def test_generate_face_shape(self):
        """Test face shape generation."""
        generator = SyntheticFaceGenerator()
        face = generator.generate_face_shape(200, 250)
        
        assert face.shape == (250, 200, 3)
        assert face.dtype == np.uint8
    
    def test_generate_synthetic_face(self):
        """Test complete synthetic face generation."""
        generator = SyntheticFaceGenerator()
        face = generator.generate_synthetic_face(1, 200, 250)
        
        assert face.shape == (250, 200, 3)
        assert face.dtype == np.uint8
    
    def test_generate_dataset(self):
        """Test dataset generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = SyntheticFaceGenerator(temp_dir)
            files = generator.generate_dataset(3)
            
            assert len(files) == 3
            for file in files:
                assert file.exists()


class TestFaceLandmarkGenerator:
    """Test cases for FaceLandmarkGenerator class."""
    
    def test_landmark_generator_initialization(self):
        """Test landmark generator initialization."""
        generator = FaceLandmarkGenerator()
        assert generator is not None
    
    def test_generate_landmarks(self):
        """Test landmark generation."""
        generator = FaceLandmarkGenerator()
        landmarks = generator.generate_landmarks((250, 200))
        
        assert landmarks.shape == (68, 2)
        assert landmarks.dtype == np.int32


class TestIntegration:
    """Integration tests."""
    
    def test_synthetic_face_swap(self):
        """Test face swapping with synthetic faces."""
        # This test would require the actual model file
        # For now, we'll test the workflow without the actual swapping
        
        # Generate synthetic faces
        generator = SyntheticFaceGenerator()
        source_face = generator.generate_synthetic_face(1)
        target_face = generator.generate_synthetic_face(2)
        
        assert source_face is not None
        assert target_face is not None
        assert source_face.shape == target_face.shape
    
    def test_configuration_workflow(self):
        """Test configuration management workflow."""
        config_manager = ConfigManager()
        
        # Test configuration loading and saving
        original_config = config_manager.get_config()
        
        # Update configuration
        config_manager.update_config(log_level="DEBUG")
        
        # Verify update
        updated_config = config_manager.get_config()
        assert updated_config.log_level == "DEBUG"
        assert updated_config.log_level != original_config.log_level


if __name__ == "__main__":
    pytest.main([__file__])
