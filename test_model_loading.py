#!/usr/bin/env python3
"""
Test script to show exactly how YOLO models are loaded
"""

import os
import time
from ultralytics import YOLO

def test_model_loading():
    print("🔍 Testing YOLO Model Loading Process")
    print("=" * 50)
    
    # Check local file
    local_file = "yolov8x.pt"
    print(f"📁 Local file check:")
    print(f"  File exists: {os.path.exists(local_file)}")
    if os.path.exists(local_file):
        size_mb = os.path.getsize(local_file) / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")
    
    print(f"\n🚀 Loading model with YOLO('yolov8x.pt')...")
    start_time = time.time()
    
    try:
        # Load the model
        model = YOLO(local_file)
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded successfully!")
        print(f"⏱️  Load time: {load_time:.2f} seconds")
        
        # Check model details
        print(f"\n📊 Model Details:")
        print(f"  Model type: {type(model)}")
        print(f"  Device: {model.device}")
        
        # Check if it's using local file or downloaded
        if hasattr(model.model, 'ckpt_path'):
            print(f"  Checkpoint path: {model.model.ckpt_path}")
        else:
            print(f"  Checkpoint path: Not accessible")
        
        # Test inference on a small dummy image
        print(f"\n🧪 Testing inference...")
        import numpy as np
        
        # Create a dummy image (1x3x640x640)
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        start_inference = time.time()
        results = model(dummy_img, verbose=False)
        inference_time = time.time() - start_inference
        
        print(f"✅ Inference successful!")
        print(f"⏱️  Inference time: {inference_time:.2f} seconds")
        print(f"📊 Results shape: {len(results)}")
        
        if len(results) > 0:
            print(f"  First result boxes: {len(results[0].boxes)}")
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    test_model_loading()
