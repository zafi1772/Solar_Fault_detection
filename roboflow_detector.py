import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from roboflow import Roboflow
import glob

class RoboflowSolarDetector:
    def __init__(self, api_key="L7zNq86EEg8nzZVQBDWK"):
        self.api_key = api_key
        self.rf = Roboflow(api_key=api_key)
        
        # Load both models
        self.thermal_model = self.rf.workspace().project("thermal-imaging-of-solar-panels-5crbd").version(1).model
        self.fault_model = self.rf.workspace().project("solar-panel-faulty-detection-a2srr").version(2).model
        
        print("✅ Both Roboflow models loaded successfully!")
    
    def analyze_image(self, image_path):
        """Analyze image using both models"""
        print(f"🔍 Analyzing: {os.path.basename(image_path)}")
        
        # Thermal analysis - higher confidence for better accuracy
        thermal_result = self.thermal_model.predict(image_path, confidence=60, overlap=30)
        
        # Fault detection - higher confidence for better accuracy
        fault_result = self.fault_model.predict(image_path, confidence=60, overlap=30)
        
        return thermal_result, fault_result
    
    def process_folder(self, input_folder, output_folder, max_images=None):
        """Process all images in folder"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Get all images
        image_files = glob.glob(os.path.join(input_folder, "*.jpg")) + \
                     glob.glob(os.path.join(input_folder, "*.jpeg")) + \
                     glob.glob(os.path.join(input_folder, "*.png")) + \
                     glob.glob(os.path.join(input_folder, "*.JPG"))
        
        # Limit number of images if specified
        if max_images:
            image_files = image_files[:max_images]
            print(f"Processing first {len(image_files)} images (limited)")
        else:
            print(f"Found {len(image_files)} images")
        
        successful = 0
        failed = 0
        
        for img_path in image_files:
            try:
                print(f"Processing: {os.path.basename(img_path)}")
                
                # Analyze image
                thermal_result, fault_result = self.analyze_image(img_path)
                
                # Save results
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                
                # Save thermal result
                thermal_result.save(os.path.join(output_folder, f"{base_name}_thermal.jpg"))
                
                # Save fault result  
                fault_result.save(os.path.join(output_folder, f"{base_name}_fault.jpg"))
                
                print(f"  ✅ Saved results for {base_name}")
                successful += 1
                
            except Exception as e:
                print(f"  ❌ Failed to process {os.path.basename(img_path)}: {str(e)}")
                failed += 1
                continue
        
        print(f"🎉 Processing completed!")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📁 Results saved to: {output_folder}")
    
    def test_single_image(self, image_path):
        """Test detection on a single image for accuracy verification"""
        print(f"🧪 Testing accuracy on: {os.path.basename(image_path)}")
        
        try:
            # Run both models
            thermal_result, fault_result = self.analyze_image(image_path)
            
            # Process and display results
            print("\n🔥 Thermal Analysis Results:")
            self._display_predictions(thermal_result)
            
            print("\n⚠️ Fault Detection Results:")
            self._display_predictions(fault_result)
            
            return thermal_result, fault_result
            
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            return None, None
    
    def _display_predictions(self, result):
        """Display predictions in a readable format"""
        try:
            if hasattr(result, 'json') and 'predictions' in result.json():
                predictions = result.json()['predictions']
                if predictions:
                    for i, pred in enumerate(predictions, 1):
                        confidence = pred.get('confidence', 0)
                        class_name = pred.get('class', 'Unknown')
                        print(f"  {i}. {class_name}: {confidence:.1%} confidence")
                else:
                    print("  No detections found")
            else:
                print("  Unable to parse results")
        except Exception as e:
            print(f"  Error displaying predictions: {str(e)}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Roboflow Solar Panel Detector')
    parser.add_argument('--api-key', default='L7zNq86EEg8nzZVQBDWK', help='Your Roboflow API key (default: built-in key)')
    parser.add_argument('--folder', help='Input folder with images')
    parser.add_argument('--output', default='roboflow_results', help='Output folder')
    parser.add_argument('--limit', type=int, help='Limit number of images to process (for testing)')
    parser.add_argument('--test', help='Test single image for accuracy verification')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = RoboflowSolarDetector(args.api_key)
    
    # Initialize detector
    detector = RoboflowSolarDetector(args.api_key)
    
    if args.test:
        # Test single image for accuracy
        detector.test_single_image(args.test)
    elif args.folder:
        # Process folder
        detector.process_folder(args.folder, args.output, args.limit)
    else:
        print("❌ Please specify either --test <image_path> or --folder <folder_path>")
        print("Example:")
        print("  python roboflow_detector.py --test 'Faulty_solar_panel/Clean/Clean (1).jpeg'")
        print("  python roboflow_detector.py --folder 'Faulty_solar_panel/Clean' --output 'results' --limit 5")

if __name__ == "__main__":
    main()
