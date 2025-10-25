"""
Command Line Interface for the face swapping application.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional
import cv2

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from face_swapper import FaceSwapper, FaceSwapConfig
from config_manager import config_manager
from data_generator import SyntheticFaceGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Modern Face Swapping Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic face swap
  python cli.py --source face1.jpg --target face2.jpg --output result.jpg
  
  # Face swap with custom settings
  python cli.py --source face1.jpg --target face2.jpg --output result.jpg --width 600 --method mixed
  
  # Generate synthetic faces
  python cli.py --generate-synthetic --count 5 --output-dir data/synthetic
  
  # Batch processing
  python cli.py --batch --source-dir source_images/ --target-dir target_images/ --output-dir results/
        """
    )
    
    # Main operation modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--source", "-s",
        type=str,
        help="Path to source face image"
    )
    group.add_argument(
        "--generate-synthetic", "-g",
        action="store_true",
        help="Generate synthetic faces for testing"
    )
    group.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Batch processing mode"
    )
    
    # Required arguments for face swapping
    parser.add_argument(
        "--target", "-t",
        type=str,
        help="Path to target face image"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path to output image"
    )
    
    # Processing options
    parser.add_argument(
        "--width", "-w",
        type=int,
        default=500,
        help="Resize width for processing (default: 500)"
    )
    parser.add_argument(
        "--method", "-m",
        choices=["normal", "mixed", "monochrome"],
        default="normal",
        help="Cloning method (default: normal)"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.5,
        help="Face detection confidence threshold (default: 0.5)"
    )
    
    # Synthetic data generation options
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=5,
        help="Number of synthetic faces to generate (default: 5)"
    )
    parser.add_argument(
        "--output-dir", "-d",
        type=str,
        default="data/synthetic",
        help="Output directory for generated faces (default: data/synthetic)"
    )
    
    # Batch processing options
    parser.add_argument(
        "--source-dir",
        type=str,
        help="Directory containing source images for batch processing"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        help="Directory containing target images for batch processing"
    )
    
    # General options
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output except errors"
    )
    
    return parser


def setup_logging(verbose: bool, quiet: bool) -> None:
    """Setup logging configuration."""
    if quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)


def get_clone_method(method: str) -> int:
    """Convert method string to OpenCV constant."""
    method_map = {
        "normal": cv2.NORMAL_CLONE,
        "mixed": cv2.MIXED_CLONE,
        "monochrome": cv2.MONOCHROME_TRANSFER
    }
    return method_map.get(method, cv2.NORMAL_CLONE)


def perform_face_swap(source_path: str, target_path: str, output_path: str, 
                     width: int, method: str, confidence: float) -> bool:
    """
    Perform face swapping operation.
    
    Args:
        source_path: Path to source image.
        target_path: Path to target image.
        output_path: Path to output image.
        width: Resize width.
        method: Cloning method.
        confidence: Detection confidence threshold.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        # Create configuration
        config = FaceSwapConfig(
            resize_width=width,
            clone_method=get_clone_method(method),
            confidence_threshold=confidence
        )
        
        # Initialize face swapper
        logger.info("Initializing face swapper...")
        swapper = FaceSwapper(config)
        
        # Perform face swap
        logger.info(f"Swapping faces: {source_path} -> {target_path}")
        result = swapper.swap_faces(source_path, target_path)
        
        if result is not None:
            # Save result
            success = swapper.save_result(result, output_path)
            if success:
                logger.info(f"Face swap completed successfully! Result saved to: {output_path}")
                return True
            else:
                logger.error("Failed to save result")
                return False
        else:
            logger.error("Face swap failed - no result generated")
            return False
            
    except Exception as e:
        logger.error(f"Error during face swap: {e}")
        return False


def generate_synthetic_faces(count: int, output_dir: str) -> bool:
    """
    Generate synthetic faces for testing.
    
    Args:
        count: Number of faces to generate.
        output_dir: Output directory.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        logger.info(f"Generating {count} synthetic faces in {output_dir}")
        
        generator = SyntheticFaceGenerator(output_dir)
        files = generator.generate_dataset(count)
        
        logger.info(f"Successfully generated {len(files)} synthetic faces:")
        for file in files:
            logger.info(f"  - {file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating synthetic faces: {e}")
        return False


def batch_process(source_dir: str, target_dir: str, output_dir: str, 
                 width: int, method: str, confidence: float) -> bool:
    """
    Process multiple images in batch mode.
    
    Args:
        source_dir: Directory containing source images.
        target_dir: Directory containing target images.
        output_dir: Output directory for results.
        width: Resize width.
        method: Cloning method.
        confidence: Detection confidence threshold.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        source_path = Path(source_dir)
        target_path = Path(target_dir)
        output_path = Path(output_dir)
        
        if not source_path.exists():
            logger.error(f"Source directory does not exist: {source_dir}")
            return False
        
        if not target_path.exists():
            logger.error(f"Target directory does not exist: {target_dir}")
            return False
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        source_files = [f for f in source_path.iterdir() 
                       if f.suffix.lower() in image_extensions]
        target_files = [f for f in target_path.iterdir() 
                       if f.suffix.lower() in image_extensions]
        
        if not source_files:
            logger.error(f"No image files found in source directory: {source_dir}")
            return False
        
        if not target_files:
            logger.error(f"No image files found in target directory: {target_dir}")
            return False
        
        logger.info(f"Found {len(source_files)} source images and {len(target_files)} target images")
        
        # Process each combination
        success_count = 0
        total_count = len(source_files) * len(target_files)
        
        for source_file in source_files:
            for target_file in target_files:
                output_file = output_path / f"{source_file.stem}_to_{target_file.stem}.jpg"
                
                logger.info(f"Processing: {source_file.name} -> {target_file.name}")
                
                if perform_face_swap(str(source_file), str(target_file), str(output_file),
                                   width, method, confidence):
                    success_count += 1
        
        logger.info(f"Batch processing completed: {success_count}/{total_count} successful")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Error during batch processing: {e}")
        return False


def main():
    """Main CLI function."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.quiet)
    
    # Load configuration if provided
    if args.config:
        config_manager.load_config()
    
    success = False
    
    try:
        if args.generate_synthetic:
            # Generate synthetic faces
            success = generate_synthetic_faces(args.count, args.output_dir)
            
        elif args.batch:
            # Batch processing
            if not args.source_dir or not args.target_dir:
                logger.error("Batch processing requires --source-dir and --target-dir")
                return 1
            
            output_dir = args.output_dir or "output"
            success = batch_process(args.source_dir, args.target_dir, output_dir,
                                 args.width, args.method, args.confidence)
            
        else:
            # Single face swap
            if not args.target or not args.output:
                logger.error("Face swapping requires --target and --output arguments")
                return 1
            
            success = perform_face_swap(args.source, args.target, args.output,
                                      args.width, args.method, args.confidence)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
