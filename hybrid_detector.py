#!/usr/bin/env python3
"""
Hybrid Solar Panel Fault Detection System
Combines YOLO (local) and Roboflow (cloud) for optimal performance
"""

import os
import time
from typing import Dict, List, Tuple, Optional
from roboflow_detector import RoboflowSolarDetector
from yolo_detector import YOLOSolarDetector

class HybridSolarDetector:
    def __init__(self, yolo_model_path="best.pt", roboflow_api_key=None, 
                 confidence=0.5, device="auto", fallback_to_roboflow=True):
        """
        Initialize hybrid detector with both YOLO and Roboflow
        
        Args:
            yolo_model_path (str): Path to trained YOLO model
            roboflow_api_key (str): Roboflow API key (optional)
            confidence (float): Default confidence threshold
            device (str): Device for YOLO inference
            fallback_to_roboflow (bool): Whether to fallback to Roboflow if YOLO fails
        """
        self.confidence = confidence
        self.fallback_to_roboflow = fallback_to_roboflow
        
        # Initialize YOLO detector
        try:
            self.yolo_detector = YOLOSolarDetector(yolo_model_path, confidence, device)
            self.yolo_available = True
            print("✅ YOLO detector initialized successfully!")
        except Exception as e:
            print(f"⚠️ YOLO detector failed to initialize: {str(e)}")
            self.yolo_detector = None
            self.yolo_available = False
        
        # Initialize Roboflow detector (if API key provided)
        if roboflow_api_key:
            try:
                self.roboflow_detector = RoboflowSolarDetector(roboflow_api_key)
                self.roboflow_available = True
                print("✅ Roboflow detector initialized successfully!")
            except Exception as e:
                print(f"⚠️ Roboflow detector failed to initialize: {str(e)}")
                self.roboflow_detector = None
                self.roboflow_available = False
        else:
            self.roboflow_detector = None
            self.roboflow_available = False
        
        if not self.yolo_available and not self.roboflow_available:
            raise Exception("No detectors available! Please check your setup.")
        
        print(f"🔧 Hybrid detector initialized:")
        print(f"  YOLO: {'✅ Available' if self.yolo_available else '❌ Not available'}")
        print(f"  Roboflow: {'✅ Available' if self.roboflow_available else '❌ Not available'}")
    
    def analyze_image(self, image_path: str, method: str = "auto", 
                     confidence: Optional[float] = None) -> Dict:
        """
        Analyze image using specified or best available method
        
        Args:
            image_path (str): Path to input image
            method (str): Detection method ('yolo', 'roboflow', 'hybrid', or 'auto')
            confidence (float): Confidence threshold override
            
        Returns:
            dict: Detection results with metadata
        """
        if confidence is None:
            confidence = self.confidence
        
        print(f"🔍 Hybrid analyzing: {os.path.basename(image_path)}")
        print(f"  Method: {method}")
        print(f"  Confidence: {confidence}")
        
        start_time = time.time()
        
        # Determine detection method
        if method == "auto":
            method = self._select_best_method()
        
        # Run detection based on method
        if method == "yolo" and self.yolo_available:
            result = self._run_yolo_detection(image_path, confidence)
        elif method == "roboflow" and self.roboflow_available:
            result = self._run_roboflow_detection(image_path, confidence)
        elif method == "hybrid" and self.yolo_available and self.roboflow_available:
            result = self._run_hybrid_detection(image_path, confidence)
        else:
            # Fallback logic
            result = self._run_fallback_detection(image_path, confidence)
        
        total_time = time.time() - start_time
        result['total_time'] = total_time
        result['method_used'] = method
        
        print(f"⚡ Analysis completed in {total_time:.2f}s using {method}")
        return result
    
    def _select_best_method(self) -> str:
        """Select the best available detection method"""
        if self.yolo_available and self.roboflow_available:
            return "hybrid"  # Use both for best results
        elif self.yolo_available:
            return "yolo"    # Local inference
        elif self.roboflow_available:
            return "roboflow"  # Cloud inference
        else:
            raise Exception("No detection methods available!")
    
    def _run_yolo_detection(self, image_path: str, confidence: float) -> Dict:
        """Run YOLO detection"""
        print("  🚀 Running YOLO detection...")
        try:
            result = self.yolo_detector.analyze_image(image_path, confidence)
            result['detector_type'] = 'yolo'
            return result
        except Exception as e:
            print(f"  ❌ YOLO detection failed: {str(e)}")
            if self.fallback_to_roboflow and self.roboflow_available:
                print("  🔄 Falling back to Roboflow...")
                return self._run_roboflow_detection(image_path, confidence)
            else:
                return {
                    'detections': [],
                    'error': f"YOLO detection failed: {str(e)}",
                    'detector_type': 'yolo'
                }
    
    def _run_roboflow_detection(self, image_path: str, confidence: float) -> Dict:
        """Run Roboflow detection"""
        print("  ☁️ Running Roboflow detection...")
        try:
            thermal_result, fault_result = self.roboflow_detector.analyze_image(image_path)
            
            # Process results into standardized format
            thermal_detections = self._process_roboflow_results(thermal_result, confidence, "thermal")
            fault_detections = self._process_roboflow_results(fault_result, confidence, "fault")
            
            all_detections = thermal_detections + fault_detections
            
            return {
                'detections': all_detections,
                'thermal_detections': thermal_detections,
                'fault_detections': fault_detections,
                'detector_type': 'roboflow',
                'inference_time': 0.0  # API call time not easily measurable
            }
        except Exception as e:
            print(f"  ❌ Roboflow detection failed: {str(e)}")
            if self.fallback_to_roboflow and self.yolo_available:
                print("  🔄 Falling back to YOLO...")
                return self._run_yolo_detection(image_path, confidence)
            else:
                return {
                    'detections': [],
                    'error': f"Roboflow detection failed: {str(e)}",
                    'detector_type': 'roboflow'
                }
    
    def _run_hybrid_detection(self, image_path: str, confidence: float) -> Dict:
        """Run both YOLO and Roboflow detection and combine results"""
        print("  🔀 Running hybrid detection (YOLO + Roboflow)...")
        
        # Run both detectors in parallel (simplified sequential for now)
        yolo_result = self._run_yolo_detection(image_path, confidence)
        roboflow_result = self._run_roboflow_detection(image_path, confidence)
        
        # Combine results
        combined_detections = self._combine_detections(
            yolo_result.get('detections', []),
            roboflow_result.get('detections', [])
        )
        
        return {
            'detections': combined_detections,
            'yolo_detections': yolo_result.get('detections', []),
            'roboflow_detections': roboflow_result.get('detections', []),
            'detector_type': 'hybrid',
            'yolo_time': yolo_result.get('inference_time', 0),
            'roboflow_time': roboflow_result.get('inference_time', 0)
        }
    
    def _run_fallback_detection(self, image_path: str, confidence: float) -> Dict:
        """Run fallback detection when preferred method fails"""
        print("  🔄 Running fallback detection...")
        
        if self.yolo_available:
            return self._run_yolo_detection(image_path, confidence)
        elif self.roboflow_available:
            return self._run_roboflow_detection(image_path, confidence)
        else:
            return {
                'detections': [],
                'error': 'No detection methods available',
                'detector_type': 'none'
            }
    
    def _process_roboflow_results(self, result, confidence_threshold: float, result_type: str) -> List[Dict]:
        """Process Roboflow results into standardized format"""
        detections = []
        
        try:
            if hasattr(result, 'json') and 'predictions' in result.json():
                predictions = result.json()['predictions']
                
                for pred in predictions:
                    confidence = pred.get('confidence', 0)
                    if confidence >= confidence_threshold:
                        # Convert center coordinates to corner coordinates
                        x_center = pred.get('x', 0)
                        y_center = pred.get('y', 0)
                        width = pred.get('width', 0)
                        height = pred.get('height', 0)
                        
                        x1 = int(x_center - width/2)
                        y1 = int(y_center - height/2)
                        x2 = int(x_center + width/2)
                        y2 = int(y_center + height/2)
                        
                        if width > 0 and height > 0:
                            detection = {
                                "class": pred.get('class', 'Unknown'),
                                "confidence": confidence,
                                "bbox": [x1, y1, x2, y2],
                                "type": result_type,
                                "source": "roboflow"
                            }
                            detections.append(detection)
            
        except Exception as e:
            print(f"  ⚠️ Error processing Roboflow results: {str(e)}")
        
        return detections
    
    def _combine_detections(self, yolo_detections: List[Dict], 
                           roboflow_detections: List[Dict]) -> List[Dict]:
        """Combine detections from both sources, removing duplicates"""
        combined = []
        
        # Add YOLO detections
        for detection in yolo_detections:
            detection['source'] = 'yolo'
            combined.append(detection)
        
        # Add Roboflow detections (avoid duplicates)
        for roboflow_det in roboflow_detections:
            is_duplicate = False
            
            for existing_det in combined:
                # Check if detections overlap significantly
                if self._detections_overlap(roboflow_det, existing_det):
                    # Keep the one with higher confidence
                    if roboflow_det['confidence'] > existing_det['confidence']:
                        # Replace existing detection
                        combined.remove(existing_det)
                        roboflow_det['source'] = 'roboflow'
                        combined.append(roboflow_det)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                roboflow_det['source'] = 'roboflow'
                combined.append(roboflow_det)
        
        return combined
    
    def _detections_overlap(self, det1: Dict, det2: Dict, threshold: float = 0.5) -> bool:
        """Check if two detections overlap significantly"""
        try:
            x1_1, y1_1, x2_1, y2_1 = det1['bbox']
            x1_2, y1_2, x2_2, y2_2 = det2['bbox']
            
            # Calculate intersection
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x1_i >= x2_i or y1_i >= y2_i:
                return False  # No intersection
            
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            
            # Calculate IoU
            union = area1 + area2 - intersection
            iou = intersection / union if union > 0 else 0
            
            return iou > threshold
            
        except Exception:
            return False
    
    def get_detector_status(self) -> Dict:
        """Get status of all available detectors"""
        status = {
            'yolo': {
                'available': self.yolo_available,
                'device': self.yolo_detector.device if self.yolo_available else None,
                'model_path': getattr(self.yolo_detector.model, 'ckpt_path', None) if self.yolo_available else None
            },
            'roboflow': {
                'available': self.roboflow_available,
                'api_key': bool(self.roboflow_detector is not None)
            },
            'fallback_enabled': self.fallback_to_roboflow,
            'confidence_threshold': self.confidence
        }
        return status
    
    def test_detectors(self, image_path: str) -> Dict:
        """Test all available detectors on a single image"""
        print(f"🧪 Testing all detectors on: {os.path.basename(image_path)}")
        
        results = {}
        
        # Test YOLO
        if self.yolo_available:
            print("\n🔍 Testing YOLO detector...")
            try:
                yolo_result = self.yolo_detector.test_single_image(image_path, self.confidence)
                results['yolo'] = yolo_result
            except Exception as e:
                results['yolo'] = {'error': str(e)}
        
        # Test Roboflow
        if self.roboflow_available:
            print("\n🔍 Testing Roboflow detector...")
            try:
                roboflow_result = self.roboflow_detector.test_single_image(image_path)
                results['roboflow'] = roboflow_result
            except Exception as e:
                results['roboflow'] = {'error': str(e)}
        
        # Test Hybrid
        if self.yolo_available and self.roboflow_available:
            print("\n🔍 Testing Hybrid detector...")
            try:
                hybrid_result = self.analyze_image(image_path, method="hybrid")
                results['hybrid'] = hybrid_result
            except Exception as e:
                results['hybrid'] = {'error': str(e)}
        
        return results

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hybrid Solar Panel Fault Detector')
    parser.add_argument('--yolo-model', default='best.pt', help='Path to YOLO model (.pt file)')
    parser.add_argument('--roboflow-key', help='Roboflow API key')
    parser.add_argument('--method', choices=['yolo', 'roboflow', 'hybrid', 'auto'], 
                       default='auto', help='Detection method to use')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', default='auto', help='Device for YOLO inference')
    parser.add_argument('--test', help='Test single image with all detectors')
    parser.add_argument('--folder', help='Process folder of images')
    parser.add_argument('--output', default='hybrid_results', help='Output folder')
    
    args = parser.parse_args()
    
    # Initialize hybrid detector
    try:
        detector = HybridSolarDetector(
            yolo_model_path=args.yolo_model,
            roboflow_api_key=args.roboflow_key,
            confidence=args.confidence,
            device=args.device
        )
        
        print(f"\n🔧 Detector Status:")
        status = detector.get_detector_status()
        for detector_name, info in status.items():
            if detector_name != 'fallback_enabled' and detector_name != 'confidence_threshold':
                print(f"  {detector_name.capitalize()}: {'✅ Available' if info['available'] else '❌ Not available'}")
        
        print(f"  Fallback: {'✅ Enabled' if status['fallback_enabled'] else '❌ Disabled'}")
        print(f"  Confidence: {status['confidence_threshold']}")
        
    except Exception as e:
        print(f"❌ Failed to initialize hybrid detector: {str(e)}")
        return
    
    if args.test:
        # Test all detectors
        results = detector.test_detectors(args.test)
        print(f"\n📊 Test Results Summary:")
        for method, result in results.items():
            if 'error' in result:
                print(f"  {method.capitalize()}: ❌ {result['error']}")
            else:
                print(f"  {method.capitalize()}: ✅ {len(result.get('detections', []))} detections")
    
    elif args.folder:
        # Process folder
        print(f"\n📁 Processing folder: {args.folder}")
        # Implementation for folder processing would go here
        print("Folder processing not yet implemented in hybrid mode")
    
    else:
        print("❌ Please specify either --test <image_path> or --folder <folder_path>")
        print("Example:")
        print("  python hybrid_detector.py --test 'Faulty_solar_panel/Clean/Clean (1).jpeg'")
        print("  python hybrid_detector.py --method hybrid --confidence 0.7")

if __name__ == "__main__":
    main()
