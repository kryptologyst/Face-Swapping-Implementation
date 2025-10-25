"""
Synthetic data generation for face swapping testing and demonstration.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import logging
from PIL import Image, ImageDraw, ImageFont
import random
import math

logger = logging.getLogger(__name__)


class SyntheticFaceGenerator:
    """Generates synthetic face images for testing face swapping."""
    
    def __init__(self, output_dir: str = "data/synthetic"):
        """
        Initialize the synthetic face generator.
        
        Args:
            output_dir: Directory to save generated images.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_face_shape(self, width: int = 200, height: int = 250) -> np.ndarray:
        """
        Generate a basic face shape.
        
        Args:
            width: Width of the face.
            height: Height of the face.
            
        Returns:
            Face shape as numpy array.
        """
        # Create a base face shape (oval)
        face = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Face color (skin tone)
        skin_color = (220, 180, 140)  # Light brown
        cv2.ellipse(face, (width//2, height//2), (width//2-10, height//2-20), 
                   0, 0, 360, skin_color, -1)
        
        return face
    
    def add_eyes(self, face: np.ndarray, eye_color: Tuple[int, int, int] = (50, 50, 50)) -> np.ndarray:
        """
        Add eyes to the face.
        
        Args:
            face: Face image array.
            eye_color: Color of the eyes.
            
        Returns:
            Face with eyes added.
        """
        height, width = face.shape[:2]
        
        # Left eye
        left_eye_center = (width//2 - 30, height//2 - 20)
        cv2.circle(face, left_eye_center, 8, eye_color, -1)
        cv2.circle(face, left_eye_center, 3, (255, 255, 255), -1)  # Eye highlight
        
        # Right eye
        right_eye_center = (width//2 + 30, height//2 - 20)
        cv2.circle(face, right_eye_center, 8, eye_color, -1)
        cv2.circle(face, right_eye_center, 3, (255, 255, 255), -1)  # Eye highlight
        
        return face
    
    def add_nose(self, face: np.ndarray) -> np.ndarray:
        """
        Add a nose to the face.
        
        Args:
            face: Face image array.
            
        Returns:
            Face with nose added.
        """
        height, width = face.shape[:2]
        
        # Simple nose
        nose_points = np.array([
            [width//2, height//2 - 10],
            [width//2 - 5, height//2 + 10],
            [width//2 + 5, height//2 + 10]
        ], np.int32)
        
        cv2.fillPoly(face, [nose_points], (200, 160, 120))  # Slightly darker skin tone
        
        return face
    
    def add_mouth(self, face: np.ndarray, smile: bool = True) -> np.ndarray:
        """
        Add a mouth to the face.
        
        Args:
            face: Face image array.
            smile: Whether to create a smiling mouth.
            
        Returns:
            Face with mouth added.
        """
        height, width = face.shape[:2]
        
        if smile:
            # Smiling mouth
            cv2.ellipse(face, (width//2, height//2 + 30), (15, 8), 0, 0, 180, (150, 100, 100), 2)
        else:
            # Neutral mouth
            cv2.line(face, (width//2 - 15, height//2 + 30), (width//2 + 15, height//2 + 30), (150, 100, 100), 2)
        
        return face
    
    def add_hair(self, face: np.ndarray, hair_color: Tuple[int, int, int] = (100, 50, 0)) -> np.ndarray:
        """
        Add hair to the face.
        
        Args:
            face: Face image array.
            hair_color: Color of the hair.
            
        Returns:
            Face with hair added.
        """
        height, width = face.shape[:2]
        
        # Hair shape (semi-circle on top)
        hair_points = np.array([
            [width//2 - 40, height//2 - 20],
            [width//2 - 50, height//2 - 40],
            [width//2, height//2 - 50],
            [width//2 + 50, height//2 - 40],
            [width//2 + 40, height//2 - 20]
        ], np.int32)
        
        cv2.fillPoly(face, [hair_points], hair_color)
        
        return face
    
    def generate_synthetic_face(self, face_id: int, width: int = 200, height: int = 250) -> np.ndarray:
        """
        Generate a complete synthetic face.
        
        Args:
            face_id: Unique identifier for the face.
            width: Width of the face.
            height: Height of the face.
            
        Returns:
            Complete synthetic face image.
        """
        # Set random seed for reproducible variations
        random.seed(face_id)
        
        # Generate base face
        face = self.generate_face_shape(width, height)
        
        # Add facial features
        face = self.add_eyes(face, eye_color=(random.randint(30, 80), random.randint(30, 80), random.randint(30, 80)))
        face = self.add_nose(face)
        face = self.add_mouth(face, smile=random.choice([True, False]))
        
        # Add hair with random color
        hair_colors = [(100, 50, 0), (50, 25, 0), (150, 100, 50), (200, 150, 100)]
        face = self.add_hair(face, hair_color=random.choice(hair_colors))
        
        return face
    
    def generate_dataset(self, num_faces: int = 10) -> List[Path]:
        """
        Generate a dataset of synthetic faces.
        
        Args:
            num_faces: Number of faces to generate.
            
        Returns:
            List of paths to generated face images.
        """
        generated_files = []
        
        for i in range(num_faces):
            face = self.generate_synthetic_face(i)
            
            # Save the face
            filename = f"synthetic_face_{i:03d}.jpg"
            filepath = self.output_dir / filename
            
            cv2.imwrite(str(filepath), face)
            generated_files.append(filepath)
            
            logger.info(f"Generated synthetic face: {filepath}")
        
        logger.info(f"Generated {num_faces} synthetic faces in {self.output_dir}")
        return generated_files


class FaceLandmarkGenerator:
    """Generates synthetic facial landmarks for testing."""
    
    def __init__(self):
        """Initialize the landmark generator."""
        pass
    
    def generate_landmarks(self, face_shape: Tuple[int, int]) -> np.ndarray:
        """
        Generate synthetic facial landmarks.
        
        Args:
            face_shape: Shape of the face image (height, width).
            
        Returns:
            Array of 68 facial landmarks.
        """
        height, width = face_shape[:2]
        
        # Generate 68 landmarks based on typical face proportions
        landmarks = []
        
        # Jaw line (0-16)
        for i in range(17):
            x = int(width * (0.1 + 0.8 * i / 16))
            y = int(height * (0.7 + 0.2 * math.sin(math.pi * i / 16)))
            landmarks.append([x, y])
        
        # Right eyebrow (17-21)
        for i in range(5):
            x = int(width * (0.3 + 0.2 * i / 4))
            y = int(height * (0.35 - 0.05 * i / 4))
            landmarks.append([x, y])
        
        # Left eyebrow (22-26)
        for i in range(5):
            x = int(width * (0.5 + 0.2 * i / 4))
            y = int(height * (0.35 - 0.05 * i / 4))
            landmarks.append([x, y])
        
        # Nose (27-35)
        nose_points = [
            (0.5, 0.4), (0.48, 0.45), (0.52, 0.45),  # Bridge
            (0.5, 0.5), (0.48, 0.55), (0.52, 0.55),  # Tip
            (0.46, 0.52), (0.54, 0.52), (0.5, 0.58)   # Nostrils
        ]
        for x_ratio, y_ratio in nose_points:
            landmarks.append([int(width * x_ratio), int(height * y_ratio)])
        
        # Right eye (36-41)
        eye_points = [
            (0.35, 0.4), (0.38, 0.38), (0.42, 0.38), (0.45, 0.4),
            (0.42, 0.42), (0.38, 0.42)
        ]
        for x_ratio, y_ratio in eye_points:
            landmarks.append([int(width * x_ratio), int(height * y_ratio)])
        
        # Left eye (42-47)
        eye_points = [
            (0.55, 0.4), (0.58, 0.38), (0.62, 0.38), (0.65, 0.4),
            (0.62, 0.42), (0.58, 0.42)
        ]
        for x_ratio, y_ratio in eye_points:
            landmarks.append([int(width * x_ratio), int(height * y_ratio)])
        
        # Outer mouth (48-59)
        mouth_outer = [
            (0.4, 0.7), (0.42, 0.72), (0.45, 0.73), (0.5, 0.74),
            (0.55, 0.73), (0.58, 0.72), (0.6, 0.7), (0.58, 0.68),
            (0.55, 0.67), (0.5, 0.66), (0.45, 0.67), (0.42, 0.68)
        ]
        for x_ratio, y_ratio in mouth_outer:
            landmarks.append([int(width * x_ratio), int(height * y_ratio)])
        
        # Inner mouth (60-67)
        mouth_inner = [
            (0.45, 0.7), (0.47, 0.71), (0.5, 0.72), (0.53, 0.71),
            (0.55, 0.7), (0.53, 0.69), (0.5, 0.68), (0.47, 0.69)
        ]
        for x_ratio, y_ratio in mouth_inner:
            landmarks.append([int(width * x_ratio), int(height * y_ratio)])
        
        return np.array(landmarks)


def create_sample_dataset():
    """Create a sample dataset for testing."""
    generator = SyntheticFaceGenerator()
    files = generator.generate_dataset(5)
    
    print(f"Created {len(files)} synthetic face images:")
    for file in files:
        print(f"  - {file}")
    
    return files


if __name__ == "__main__":
    create_sample_dataset()
