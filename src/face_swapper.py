"""
Modern Face Swapping Implementation

This module provides a modernized face swapping implementation using OpenCV and Dlib,
with enhanced error handling, type hints, and support for multiple face detection methods.
"""

import cv2
import dlib
import numpy as np
import logging
from typing import Optional, Tuple, List, Union
from pathlib import Path
import imutils
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FaceSwapConfig:
    """Configuration class for face swapping parameters."""
    predictor_path: str = "models/shape_predictor_68_face_landmarks.dat"
    resize_width: int = 500
    clone_method: int = cv2.NORMAL_CLONE
    border_mode: int = cv2.BORDER_REFLECT_101
    confidence_threshold: float = 0.5


class FaceDetector:
    """Modern face detector using Dlib with enhanced error handling."""
    
    def __init__(self, predictor_path: str) -> None:
        """
        Initialize the face detector.
        
        Args:
            predictor_path: Path to the Dlib shape predictor model file.
            
        Raises:
            FileNotFoundError: If the predictor file doesn't exist.
            RuntimeError: If Dlib initialization fails.
        """
        self.predictor_path = Path(predictor_path)
        if not self.predictor_path.exists():
            raise FileNotFoundError(f"Predictor file not found: {predictor_path}")
        
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(str(self.predictor_path))
            logger.info("Face detector initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize face detector: {e}")
    
    def get_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract facial landmarks from an image.
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            Array of landmark points or None if no face detected.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 1)
            
            if len(faces) == 0:
                logger.warning("No faces detected in image")
                return None
            
            # Use the first detected face
            face = faces[0]
            landmarks = self.predictor(gray, face)
            
            # Convert to numpy array
            points = np.array([[p.x, p.y] for p in landmarks.parts()])
            logger.info(f"Detected {len(points)} facial landmarks")
            return points
            
        except Exception as e:
            logger.error(f"Error extracting landmarks: {e}")
            return None


class FaceSwapper:
    """Modern face swapper with enhanced blending and error handling."""
    
    def __init__(self, config: FaceSwapConfig) -> None:
        """
        Initialize the face swapper.
        
        Args:
            config: Configuration object for face swapping parameters.
        """
        self.config = config
        self.detector = FaceDetector(config.predictor_path)
    
    def load_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load and preprocess an image.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Preprocessed image or None if loading fails.
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                logger.error(f"Image file not found: {image_path}")
                return None
            
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Resize for easier processing
            image = imutils.resize(image, width=self.config.resize_width)
            logger.info(f"Loaded and resized image: {image_path}")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def _find_triangle_indices(self, points: List[np.ndarray], hull_indices: np.ndarray) -> List[List[int]]:
        """
        Find triangle indices for Delaunay triangulation.
        
        Args:
            points: List of hull points.
            hull_indices: Convex hull indices.
            
        Returns:
            List of triangle indices.
        """
        def find_index(point: Tuple[float, float], points_list: List[np.ndarray]) -> int:
            """Find the index of a point in the points list."""
            for i, pt in enumerate(points_list):
                if abs(point[0] - pt[0]) < 1 and abs(point[1] - pt[1]) < 1:
                    return i
            return -1
        
        # Compute Delaunay triangulation
        size = (500, 500)  # Default size for triangulation
        rect = (0, 0, size[1], size[0])
        subdiv = cv2.Subdiv2D(rect)
        
        for p in points:
            subdiv.insert((p[0], p[1]))
        
        triangles = subdiv.getTriangleList()
        
        # Convert triangles to indices
        tri_indices = []
        for triangle in triangles:
            pts = [(triangle[0], triangle[1]), (triangle[2], triangle[3]), (triangle[4], triangle[5])]
            indices = [find_index(p, points) for p in pts]
            if -1 not in indices:
                tri_indices.append(indices)
        
        return tri_indices
    
    def _warp_triangle(self, src_img: np.ndarray, dst_img: np.ndarray, 
                      src_tri: np.ndarray, dst_tri: np.ndarray) -> np.ndarray:
        """
        Warp a triangle from source to destination.
        
        Args:
            src_img: Source image.
            dst_img: Destination image.
            src_tri: Source triangle points.
            dst_tri: Destination triangle points.
            
        Returns:
            Warped image with triangle applied.
        """
        # Compute affine transform
        matrix = cv2.getAffineTransform(src_tri, dst_tri)
        
        # Apply warp to triangle region
        rect = cv2.boundingRect(dst_tri)
        warped_triangle = cv2.warpAffine(
            src_img, matrix, (rect[2], rect[3]),
            borderMode=self.config.border_mode
        )
        
        # Create mask and insert warped triangle
        mask = np.zeros((rect[3], rect[2], 3), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dst_tri - dst_tri.min(axis=0)), (1, 1, 1), 16)
        
        # Apply the warped triangle to the destination image
        dst_img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] *= (1 - mask)
        dst_img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] += warped_triangle * mask
        
        return dst_img
    
    def swap_faces(self, src_image_path: Union[str, Path], 
                   dst_image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Perform face swapping between two images.
        
        Args:
            src_image_path: Path to source face image.
            dst_image_path: Path to destination face image.
            
        Returns:
            Result image with swapped faces or None if operation fails.
        """
        try:
            # Load images
            src_img = self.load_image(src_image_path)
            dst_img = self.load_image(dst_image_path)
            
            if src_img is None or dst_img is None:
                logger.error("Failed to load one or both images")
                return None
            
            # Get landmarks for both faces
            src_points = self.detector.get_landmarks(src_img)
            dst_points = self.detector.get_landmarks(dst_img)
            
            if src_points is None or dst_points is None:
                logger.error("Could not detect faces in one or both images")
                return None
            
            # Compute convex hull for blending
            hull_indices = cv2.convexHull(np.array(dst_points), returnPoints=False)
            src_hull = [src_points[int(i)] for i in hull_indices]
            dst_hull = [dst_points[int(i)] for i in hull_indices]
            
            # Find triangle indices
            tri_indices = self._find_triangle_indices(dst_hull, hull_indices)
            
            # Warp each triangle from source to destination
            warped_img = np.copy(dst_img)
            for tri in tri_indices:
                src_tri = np.float32([src_hull[i] for i in tri])
                dst_tri = np.float32([dst_hull[i] for i in tri])
                warped_img = self._warp_triangle(src_img, warped_img, src_tri, dst_tri)
            
            # Create mask for seamless cloning
            hull8U = [(int(p[0]), int(p[1])) for p in dst_hull]
            mask = np.zeros(dst_img.shape, dtype=dst_img.dtype)
            cv2.fillConvexPoly(mask, np.array(hull8U), (255, 255, 255))
            
            # Find center for cloning
            rect = cv2.boundingRect(np.float32([hull8U]))
            center = (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)
            
            # Use seamless cloning to blend the faces
            output = cv2.seamlessClone(warped_img, dst_img, mask, center, self.config.clone_method)
            
            logger.info("Face swapping completed successfully")
            return output
            
        except Exception as e:
            logger.error(f"Error during face swapping: {e}")
            return None
    
    def save_result(self, result: np.ndarray, output_path: Union[str, Path]) -> bool:
        """
        Save the face swap result to a file.
        
        Args:
            result: Result image array.
            output_path: Path to save the result.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            success = cv2.imwrite(str(output_path), result)
            if success:
                logger.info(f"Result saved to: {output_path}")
                return True
            else:
                logger.error(f"Failed to save result to: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving result: {e}")
            return False


def main():
    """Main function for testing the face swapper."""
    config = FaceSwapConfig()
    swapper = FaceSwapper(config)
    
    # Example usage
    src_path = "data/face1.jpg"
    dst_path = "data/face2.jpg"
    output_path = "data/result.jpg"
    
    result = swapper.swap_faces(src_path, dst_path)
    if result is not None:
        swapper.save_result(result, output_path)
        print(f"Face swap completed! Result saved to {output_path}")
    else:
        print("Face swap failed!")


if __name__ == "__main__":
    main()
