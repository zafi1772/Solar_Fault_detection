#!/usr/bin/env python3
"""
Simple YOLO test script to verify functionality
"""

import os
import sys
from PIL import Image

def test_yolo_import():
    """Test if YOLO can be imported"""
    try:
        from ultralytics import YOLO
        print("✅ YOLO imported successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to import YOLO: {str(e)}")
        return False

def test_yolo_model_loading():
    """Test if YOLO model can be loaded"""
    try:
        from ultralytics import YOLO
        
        # Try to load a simple model
        print("🔧 Loading YOLO model...")
        model = YOLO('yolov8x.pt')
        print("✅ YOLO model loaded successfully!")
        return True, model
    except Exception as e:
        print(f"❌ Failed to load YOLO model: {str(e)}")
        return False, None

def test_yolo_inference(model, image_path):
    """Test YOLO inference on an image"""
    try:
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return False
        
        print(f"🔍 Testing inference on: {image_path}")
        
        # Run inference
        results = model(image_path, verbose=False)
        
        print(f"✅ Inference completed successfully!")
        print(f"📊 Results: {len(results)} result(s)")
        
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                print(f"📦 Detections: {len(result.boxes)} object(s)")
                return True
            else:
                print("📦 No detections found (this is normal for clean images)")
                return True
        else:
            print("📦 No results returned")
            return False
            
    except Exception as e:
        print(f"❌ Inference failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🧪 YOLO Functionality Test")
    print("=" * 40)
    
    # Test 1: Import
    if not test_yolo_import():
        print("❌ YOLO import test failed!")
        return False
    
    # Test 2: Model Loading
    success, model = test_yolo_model_loading()
    if not success:
        print("❌ YOLO model loading test failed!")
        return False
    
    # Test 3: Inference
    test_image = "Faulty_solar_panel/Clean/Clean (1).jpeg"
    if os.path.exists(test_image):
        if test_yolo_inference(model, test_image):
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Inference test failed!")
            return False
    else:
        print(f"⚠️ Test image not found: {test_image}")
        print("✅ Basic functionality tests passed!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
