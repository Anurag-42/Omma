#!/usr/bin/env python3
"""
Speed test for bug detection model
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
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Deployment import BugDetectionEvaluator

def speed_test():
    MODEL_PATH = "enhanced_faster_rcnn_bug_detection_best.pth"
    TEST_IMAGE = "annotated_img/image3rd1.jpg"
    
    print("🔄 Loading model...")
    start_load = time.time()
    evaluator = BugDetectionEvaluator(MODEL_PATH)
    load_time = time.time() - start_load
    print(f"✅ Model loaded in {load_time:.2f} seconds")
    
    print(f"\n🖼️  Testing speed on: {TEST_IMAGE}")
    
    # Run multiple tests to get average
    times = []
    for i in range(5):
        start = time.time()
        result, original_image = evaluator.detect_bugs_in_image(TEST_IMAGE, conf_threshold=0.5)
        end = time.time()
        inference_time = end - start
        times.append(inference_time)
        
        if result:
            num_bugs = result['detections']['count']
            print(f"  Test {i+1}: {inference_time:.3f}s ({num_bugs} bugs)")
        else:
            print(f"  Test {i+1}: {inference_time:.3f}s (failed)")
    
    avg_time = sum(times) / len(times)
    fps = 1 / avg_time
    
    print(f"\n📊 Speed Results:")
    print(f"  Average inference time: {avg_time:.3f} seconds")
    print(f"  Images per second: {fps:.1f} FPS")
    print(f"  Model loading time: {load_time:.2f} seconds")

if __name__ == "__main__":
    speed_test()