#!/usr/bin/env python3
"""
Demonstration script for the modernized face swapping application.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_generator import SyntheticFaceGenerator
from face_swapper import FaceSwapper, FaceSwapConfig
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_synthetic_faces():
    """Demonstrate synthetic face generation."""
    print("🎨 Generating synthetic faces for demonstration...")
    
    generator = SyntheticFaceGenerator("data/synthetic")
    files = generator.generate_dataset(3)
    
    print(f"✅ Generated {len(files)} synthetic faces:")
    for file in files:
        print(f"   - {file}")
    
    return files


def demonstrate_face_swapping(source_path: str, target_path: str):
    """Demonstrate face swapping with synthetic faces."""
    print(f"🔄 Demonstrating face swapping...")
    print(f"   Source: {source_path}")
    print(f"   Target: {target_path}")
    
    try:
        # Create configuration
        config = FaceSwapConfig(
            resize_width=400,
            clone_method=cv2.NORMAL_CLONE
        )
        
        # Initialize face swapper
        swapper = FaceSwapper(config)
        
        # Perform face swap
        result = swapper.swap_faces(source_path, target_path)
        
        if result is not None:
            # Save result
            output_path = "data/synthetic/swap_result.jpg"
            success = swapper.save_result(result, output_path)
            
            if success:
                print(f"✅ Face swap completed! Result saved to: {output_path}")
                return True
            else:
                print("❌ Failed to save result")
                return False
        else:
            print("❌ Face swap failed - no result generated")
            return False
            
    except Exception as e:
        print(f"❌ Error during face swap: {e}")
        return False


def main():
    """Main demonstration function."""
    print("🚀 Face Swapping Application Demonstration")
    print("=" * 50)
    
    # Create necessary directories
    Path("data/synthetic").mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic faces
    files = demonstrate_synthetic_faces()
    
    if len(files) >= 2:
        # Demonstrate face swapping
        source_file = str(files[0])
        target_file = str(files[1])
        
        success = demonstrate_face_swapping(source_file, target_file)
        
        if success:
            print("\n🎉 Demonstration completed successfully!")
            print("\n📖 Next steps:")
            print("   1. Run the web app: streamlit run web_app/app.py")
            print("   2. Try CLI: python cli.py --help")
            print("   3. Check the generated files in data/synthetic/")
        else:
            print("\n⚠️  Demonstration completed with some issues")
            print("   This is expected without the Dlib model file")
    else:
        print("❌ Could not generate enough synthetic faces for demonstration")
    
    print("\n📚 For more information, see README.md")


if __name__ == "__main__":
    main()
