#!/usr/bin/env python3
"""
Analyze image properties to understand processing time differences
"""

from PIL import Image
import os

def analyze_image(path):
    if not os.path.exists(path):
        print(f"❌ {path} not found")
        return
    
    img = Image.open(path)
    file_size = os.path.getsize(path)
    
    print(f"\n📸 {path}:")
    print(f"  Dimensions: {img.size[0]} x {img.size[1]} pixels")
    print(f"  File size: {file_size / (1024*1024):.2f} MB")
    print(f"  Format: {img.format}")
    print(f"  Mode: {img.mode}")
    print(f"  Pixel count: {img.size[0] * img.size[1]:,}")
    
    # Check if image has transparency or other channels
    if hasattr(img, 'info'):
        print(f"  Info: {img.info}")

if __name__ == "__main__":
    analyze_image("image12.jpg")
    analyze_image("image12-min.jpg")