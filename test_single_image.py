#!/usr/bin/env python3
"""
Test single image with bug detection
"""

import ssl
import certifi
import os
import time

# Fix SSL certificate issue
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import sys
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Deployment import BugDetectionEvaluator

def test_single_image():
    MODEL_PATH = "enhanced_faster_rcnn_bug_detection_best.pth"
    TEST_IMAGE = "image12-min.jpg"
    OUTPUT_FOLDER = "single_test_output"
    CONF_THRESHOLD = 0.5
    
    # Create output directories
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "high_confidence"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "all_detections"), exist_ok=True)
    
    print("=" * 60)
    print(f"TESTING: {TEST_IMAGE}")
    print("=" * 60)
    
    # Initialize the model
    print("🔄 Loading model...")
    evaluator = BugDetectionEvaluator(MODEL_PATH)
    print("✅ Model ready!")
    
    print(f"\n🖼️  Processing: {TEST_IMAGE}")
    
    start_time = time.time()
    
    try:
        # Detect bugs
        result, original_image = evaluator.detect_bugs_in_image(TEST_IMAGE, conf_threshold=CONF_THRESHOLD)
        
        process_time = time.time() - start_time
        
        if result is None:
            print("❌ Detection failed for this image")
            return
        
        num_detections = result['detections']['count']
        num_all = result['all_detections']['count']
        
        print(f"🐛 Found {num_detections} bug(s) at confidence ≥ {CONF_THRESHOLD}")
        print(f"📊 Total predictions: {num_all} (including low confidence)")
        print(f"⚡ Processing time: {process_time:.3f} seconds")
        
        if result['detections']['scores']:
            max_conf = max(result['detections']['scores'])
            avg_conf = np.mean(result['detections']['scores'])
            min_conf = min(result['detections']['scores'])
            print(f"🎯 Confidence scores - Max: {max_conf:.3f}, Avg: {avg_conf:.3f}, Min: {min_conf:.3f}")
        
        # Save visualizations
        if num_detections > 0:
            # High confidence visualization (thin lines!)
            vis_image = evaluator.visualize_detections(
                original_image.copy(), result, 
                show_all_detections=False, 
                conf_threshold=CONF_THRESHOLD
            )
            vis_path = os.path.join(OUTPUT_FOLDER, "high_confidence", f"detected_{Path(TEST_IMAGE).name}")
            vis_image.save(vis_path)
            print(f"💾 High confidence detections saved: {vis_path}")
            
            # All detections visualization
            vis_all = evaluator.visualize_detections(
                original_image.copy(), result,
                show_all_detections=True, 
                conf_threshold=0.1
            )
            vis_all_path = os.path.join(OUTPUT_FOLDER, "all_detections", f"all_{Path(TEST_IMAGE).name}")
            vis_all.save(vis_all_path)
            print(f"💾 All detections saved: {vis_all_path}")
        else:
            # Save original if no detections
            no_detect_path = os.path.join(OUTPUT_FOLDER, "high_confidence", f"no_bugs_{Path(TEST_IMAGE).name}")
            original_image.save(no_detect_path)
            print(f"💾 No bugs detected, saved original: {no_detect_path}")
        
        print(f"\n✅ Results saved to: {OUTPUT_FOLDER}/")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_single_image()