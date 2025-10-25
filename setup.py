#!/usr/bin/env python3
"""
Setup script for the Face Swapping Application.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def create_directories() -> bool:
    """Create necessary directories."""
    directories = [
        "data",
        "data/synthetic",
        "models",
        "output",
        "temp",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    return True


def install_dependencies() -> bool:
    """Install Python dependencies."""
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    
    # Check for optional dependencies
    optional_deps = ["streamlit", "opencv-python", "dlib"]
    for dep in optional_deps:
        try:
            __import__(dep.replace("-", "_"))
            print(f"✅ {dep} installed")
        except ImportError:
            print(f"⚠️  {dep} not installed - some features may not work")
    
    return True


def download_dlib_model() -> bool:
    """Download Dlib shape predictor model."""
    model_path = Path("models/shape_predictor_68_face_landmarks.dat")
    
    if model_path.exists():
        print("✅ Dlib model already exists")
        return True
    
    print("📥 Dlib model not found. Please download it manually:")
    print("   1. Go to: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    print("   2. Extract the file")
    print("   3. Place it in the models/ directory")
    print("   4. Rename it to: shape_predictor_68_face_landmarks.dat")
    
    return False


def run_tests() -> bool:
    """Run the test suite."""
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("⚠️  Some tests may fail without the Dlib model file")
        return False
    return True


def generate_sample_data() -> bool:
    """Generate sample synthetic data."""
    try:
        from src.data_generator import create_sample_dataset
        create_sample_dataset()
        print("✅ Sample synthetic data generated")
        return True
    except Exception as e:
        print(f"⚠️  Could not generate sample data: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Face Swapping Application...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Download Dlib model
    download_dlib_model()
    
    # Generate sample data
    generate_sample_data()
    
    # Run tests
    run_tests()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\n📖 Next steps:")
    print("   1. Download the Dlib model file (see instructions above)")
    print("   2. Run the web app: streamlit run web_app/app.py")
    print("   3. Or use CLI: python cli.py --help")
    print("\n📚 For more information, see README.md")


if __name__ == "__main__":
    main()
