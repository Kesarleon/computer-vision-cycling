#!/usr/bin/env python3
"""
Simple script to check if all dependencies are properly installed
"""
import sys

def check_imports():
    """Check if all required modules can be imported"""
    try:
        import streamlit
        print("✓ Streamlit imported successfully")
    except ImportError as e:
        print(f"✗ Streamlit import failed: {e}")
        return False
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics imported successfully")
    except ImportError as e:
        print(f"✗ Ultralytics import failed: {e}")
        return False
    
    try:
        from src.tracker import CentroidTracker
        from src.video_processing import process_video
        print("✓ Local modules imported successfully")
    except ImportError as e:
        print(f"✗ Local modules import failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Checking setup for Streamlit deployment...")
    if check_imports():
        print("\n🎉 All dependencies are properly installed!")
        sys.exit(0)
    else:
        print("\n❌ Some dependencies are missing. Please check the error messages above.")
        sys.exit(1)