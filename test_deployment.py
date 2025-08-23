#!/usr/bin/env python3
"""
Test script for bug detection deployment
Tests the model on existing annotated images
"""

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import json
import numpy as np
import os
import sys
from pathlib import Path

# Import the BugDetectionEvaluator class from Deployment.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Deployment import BugDetectionEvaluator

def test_bug_detection():
    # Configuration
    MODEL_PATH = "enhanced_faster_rcnn_bug_detection_best.pth"
    TEST_IMAGE_FOLDER = "annotated_img"
    OUTPUT_FOLDER = "test_output"
    CONF_THRESHOLD = 0.5
    
    # Create output directories
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "high_confidence"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "all_detections"), exist_ok=True)
    
    print("=" * 60)
    print("BUG DETECTION MODEL TEST")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Test Images: {TEST_IMAGE_FOLDER}")
    print(f"Output: {OUTPUT_FOLDER}")
    print(f"Confidence Threshold: {CONF_THRESHOLD}")
    print("=" * 60)
    
    # Initialize the model
    print("\n🔄 Loading bug detection model...")
    try:
        evaluator = BugDetectionEvaluator(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Get test images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_files = [f for f in Path(TEST_IMAGE_FOLDER).glob("*") if f.suffix.lower() in image_extensions]
    
    # Limit to first 5 images for testing
    test_images = image_files[:5]
    
    if not test_images:
        print(f"❌ No images found in {TEST_IMAGE_FOLDER}")
        return
    
    print(f"\n📸 Found {len(image_files)} images, testing first {len(test_images)}...")
    print("-" * 40)
    
    results_summary = []
    
    for idx, image_path in enumerate(test_images, 1):
        print(f"\n[{idx}/{len(test_images)}] Processing: {image_path.name}")
        
        try:
            # Detect bugs
            result, original_image = evaluator.detect_bugs_in_image(str(image_path), conf_threshold=CONF_THRESHOLD)
            
            if result is None:
                print("  ⚠️  Detection failed for this image")
                continue
            
            num_detections = result['detections']['count']
            print(f"  🐛 Found {num_detections} bug(s) (conf ≥ {CONF_THRESHOLD})")
            
            if result['detections']['scores']:
                max_conf = max(result['detections']['scores'])
                avg_conf = np.mean(result['detections']['scores'])
                print(f"  📊 Confidence - Max: {max_conf:.3f}, Avg: {avg_conf:.3f}")
            
            # Save visualizations
            if num_detections > 0:
                # High confidence visualization
                vis_image = evaluator.visualize_detections(
                    original_image.copy(), result, 
                    show_all_detections=False, 
                    conf_threshold=CONF_THRESHOLD
                )
                vis_path = os.path.join(OUTPUT_FOLDER, "high_confidence", f"detected_{image_path.name}")
                vis_image.save(vis_path)
                print(f"  💾 Saved: {vis_path}")
                
                # All detections visualization
                vis_all = evaluator.visualize_detections(
                    original_image.copy(), result,
                    show_all_detections=True, 
                    conf_threshold=0.1
                )
                vis_all_path = os.path.join(OUTPUT_FOLDER, "all_detections", f"all_{image_path.name}")
                vis_all.save(vis_all_path)
            else:
                # Save original if no detections
                no_detect_path = os.path.join(OUTPUT_FOLDER, "high_confidence", f"no_bugs_{image_path.name}")
                original_image.save(no_detect_path)
                print(f"  💾 No bugs detected, saved original")
            
            # Store results
            results_summary.append({
                'image': image_path.name,
                'detections': num_detections,
                'max_confidence': max(result['detections']['scores']) if result['detections']['scores'] else 0
            })
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_images = len(test_images)
    successful = len(results_summary)
    total_bugs = sum(r['detections'] for r in results_summary)
    images_with_bugs = sum(1 for r in results_summary if r['detections'] > 0)
    
    print(f"📸 Images processed: {successful}/{total_images}")
    print(f"🐛 Total bugs detected: {total_bugs}")
    print(f"📊 Images with bugs: {images_with_bugs}/{successful} ({images_with_bugs/successful*100:.1f}%)")
    
    if results_summary:
        print(f"\n📋 Detection Details:")
        for r in results_summary:
            status = "✅" if r['detections'] > 0 else "⭕"
            print(f"  {status} {r['image']}: {r['detections']} bug(s)")
            if r['detections'] > 0:
                print(f"      Max confidence: {r['max_confidence']:.3f}")
    
    print(f"\n✅ Results saved to: {OUTPUT_FOLDER}/")
    print("  - high_confidence/: Detections above threshold")
    print("  - all_detections/: All predictions (including low confidence)")

if __name__ == "__main__":
    test_bug_detection()