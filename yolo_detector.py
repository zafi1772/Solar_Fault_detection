#!/usr/bin/env python3
"""
YOLO-based Solar Panel Fault Detection System
Local inference using trained YOLO models for real-time detection
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import glob
import time

class YOLOSolarDetector:
    def __init__(self, model_path="yolov8x.pt", confidence=0.5, device="auto"):
        """
        Initialize YOLO detector for solar panel fault detection
        
        Args:
            model_path (str): Path to trained YOLO model (.pt file)
            confidence (float): Default confidence threshold (0.0-1.0)
            device (str): Device to run inference on ('cpu', 'cuda', or 'auto')
        """
        self.confidence = confidence
        self.device = device
        
        # Load YOLO model
        try:
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"✅ YOLO model loaded successfully from: {model_path}")
            else:
                # Use pre-trained YOLO model as fallback
                print(f"⚠️ Model not found at {model_path}, using pre-trained YOLOv8x (best available)")
                try:
                    self.model = YOLO('yolov8x')
                except Exception as e:
                    print(f"⚠️ Failed to load yolov8x: {str(e)}")
                    print("🔧 Trying alternative approach...")
                    # Use a different approach for compatibility
                    self.model = YOLO('yolov8x', task='detect')
            
            # Set device
            if device == "auto":
                self.device = "cuda" if self.model.device.type == "cuda" else "cpu"
            else:
                self.device = device
            
            print(f"🚀 Running on device: {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {str(e)}")
            raise
        
        # Solar panel fault classes (including thermal damage)
        self.fault_classes = [
            "bird-drop", "clean", "dusty", "electrical-damage", 
            "physical-damage", "snow-covered", "thermal-damage"
        ]
        
        # Color mapping for different fault types
        self.colors = {
            "bird-drop": (255, 0, 0),      # Red
            "clean": (0, 255, 0),          # Green
            "dusty": (255, 165, 0),        # Orange
            "electrical-damage": (255, 0, 255),  # Magenta
            "physical-damage": (0, 0, 255),      # Blue
            "snow-covered": (128, 128, 128),     # Gray
            "thermal-damage": (255, 20, 147)     # Deep Pink (for thermal anomalies)
        }
    
    def analyze_image(self, image_path, confidence=None):
        """
        Analyze image for solar panel faults using YOLO
        
        Args:
            image_path (str): Path to input image
            confidence (float): Confidence threshold override
            
        Returns:
            dict: Detection results with bounding boxes and classifications
        """
        if confidence is None:
            confidence = self.confidence
            
        print(f"🔍 YOLO analyzing: {os.path.basename(image_path)}")
        start_time = time.time()
        
        try:
            # Run YOLO inference
            results = self.model(image_path, conf=confidence, verbose=False)
            
            # Process results
            detections = self._process_yolo_results(results[0], confidence)
            
            inference_time = time.time() - start_time
            print(f"⚡ YOLO inference completed in {inference_time:.2f}s")
            print(f"📊 Found {len(detections)} detections")
            
            return {
                'detections': detections,
                'inference_time': inference_time,
                'model_info': {
                    'name': os.path.basename(self.model.ckpt_path) if hasattr(self.model, 'ckpt_path') else 'YOLO',
                    'device': self.device,
                    'confidence_threshold': confidence
                }
            }
            
        except Exception as e:
            print(f"❌ YOLO analysis failed: {str(e)}")
            return {
                'detections': [],
                'error': str(e),
                'inference_time': time.time() - start_time
            }
    
    def _process_yolo_results(self, result, confidence_threshold):
        """
        Process YOLO results into standardized detection format
        
        Args:
            result: YOLO result object
            confidence_threshold (float): Minimum confidence score
            
        Returns:
            list: List of detection dictionaries
        """
        detections = []
        
        if result.boxes is None:
            return detections
        
        # Get bounding boxes, confidence scores, and class IDs
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            if conf >= confidence_threshold:
                # Get class name
                class_name = self._get_class_name(class_id)
                
                detection = {
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': [int(x) for x in box],  # [x1, y1, x2, y2]
                    'class_id': int(class_id),
                    'type': 'yolo'
                }
                
                detections.append(detection)
                print(f"  📍 {class_name}: {conf:.2%} at {detection['bbox']}")
        
        return detections
    
    def _get_class_name(self, class_id):
        """Get class name from class ID"""
        if 0 <= class_id < len(self.fault_classes):
            return self.fault_classes[class_id]
        return f"class_{class_id}"
    
    def process_folder(self, input_folder, output_folder, max_images=None, confidence=None):
        """
        Process all images in folder using YOLO
        
        Args:
            input_folder (str): Input folder path
            output_folder (str): Output folder path
            max_images (int): Maximum number of images to process
            confidence (float): Confidence threshold
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Get all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        
        # Limit number of images if specified
        if max_images:
            image_files = image_files[:max_images]
            print(f"Processing first {len(image_files)} images (limited)")
        else:
            print(f"Found {len(image_files)} images")
        
        successful = 0
        failed = 0
        total_time = 0
        
        for img_path in image_files:
            try:
                print(f"Processing: {os.path.basename(img_path)}")
                
                # Analyze image
                result = self.analyze_image(img_path, confidence)
                
                if 'error' not in result:
                    # Save annotated image
                    annotated_img = self._create_annotated_image(img_path, result['detections'])
                    
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    output_path = os.path.join(output_folder, f"{base_name}_yolo_annotated.jpg")
                    annotated_img.save(output_path)
                    
                    # Save detection results as JSON
                    import json
                    json_path = os.path.join(output_folder, f"{base_name}_yolo_results.json")
                    with open(json_path, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    print(f"  ✅ Saved results for {base_name}")
                    successful += 1
                    total_time += result['inference_time']
                else:
                    print(f"  ❌ Failed: {result['error']}")
                    failed += 1
                
            except Exception as e:
                print(f"  ❌ Failed to process {os.path.basename(img_path)}: {str(e)}")
                failed += 1
                continue
        
        avg_time = total_time / successful if successful > 0 else 0
        print(f"🎉 YOLO processing completed!")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"⏱️ Average inference time: {avg_time:.2f}s")
        print(f"📁 Results saved to: {output_folder}")
    
    def _create_annotated_image(self, image_path, detections):
        """
        Create annotated image with bounding boxes and labels
        
        Args:
            image_path (str): Path to original image
            detections (list): List of detection dictionaries
            
        Returns:
            PIL.Image: Annotated image
        """
        # Load image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            except:
                font = ImageFont.load_default()
        
        # Draw detection boxes and labels
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Get color for this class
            color = self.colors.get(class_name, (255, 255, 0))  # Yellow default
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Create label text
            label_text = f"{class_name}: {confidence:.1%}"
            
            # Calculate label dimensions
            bbox = draw.textbbox((0, 0), label_text, font=font)
            label_width = bbox[2] - bbox[0]
            label_height = bbox[3] - bbox[1]
            
            # Position label above the box
            label_x = x1
            label_y = max(0, y1 - label_height - 5)
            
            # Draw label background
            draw.rectangle([
                label_x, label_y,
                label_x + label_width + 10,
                label_y + label_height + 5
            ], fill=color)
            
            # Draw label text
            draw.text((label_x + 5, label_y + 2), label_text, fill="white", font=font)
        
        return image
    
    def test_single_image(self, image_path, confidence=None):
        """
        Test detection on a single image for accuracy verification
        
        Args:
            image_path (str): Path to test image
            confidence (float): Confidence threshold
        """
        print(f"🧪 YOLO testing accuracy on: {os.path.basename(image_path)}")
        
        try:
            result = self.analyze_image(image_path, confidence)
            
            if 'error' not in result:
                print(f"\n📊 YOLO Detection Results:")
                print(f"  Model: {result['model_info']['name']}")
                print(f"  Device: {result['model_info']['device']}")
                print(f"  Inference Time: {result['inference_time']:.2f}s")
                print(f"  Confidence Threshold: {result['model_info']['confidence_threshold']}")
                print(f"  Total Detections: {len(result['detections'])}")
                
                for i, detection in enumerate(result['detections'], 1):
                    print(f"    {i}. {detection['class']}: {detection['confidence']:.1%}")
                
                return result
            else:
                print(f"❌ Test failed: {result['error']}")
                return None
                
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            return None
    
    def get_model_info(self):
        """Get information about the loaded model"""
        info = {
            'model_type': 'YOLO',
            'device': self.device,
            'confidence_threshold': self.confidence,
            'fault_classes': self.fault_classes,
            'model_path': getattr(self.model, 'ckpt_path', 'Unknown')
        }
        return info

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO Solar Panel Fault Detector')
    parser.add_argument('--model', default='yolov8x.pt', help='Path to YOLO model (.pt file)')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--device', default='auto', help='Device to use (cpu, cuda, or auto)')
    parser.add_argument('--folder', help='Input folder with images')
    parser.add_argument('--output', default='yolo_results', help='Output folder')
    parser.add_argument('--limit', type=int, help='Limit number of images to process')
    parser.add_argument('--test', help='Test single image for accuracy verification')
    
    args = parser.parse_args()
    
    # Initialize detector
    try:
        detector = YOLOSolarDetector(args.model, args.confidence, args.device)
        print(f"🔧 Model Info: {detector.get_model_info()}")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {str(e)}")
        return
    
    if args.test:
        # Test single image
        detector.test_single_image(args.test, args.confidence)
    elif args.folder:
        # Process folder
        detector.process_folder(args.folder, args.output, args.limit, args.confidence)
    else:
        print("❌ Please specify either --test <image_path> or --folder <folder_path>")
        print("Example:")
        print("  python yolo_detector.py --test 'Faulty_solar_panel/Clean/Clean (1).jpeg'")
        print("  python yolo_detector.py --folder 'Faulty_solar_panel/Clean' --output 'yolo_results' --limit 5")

if __name__ == "__main__":
    main()
