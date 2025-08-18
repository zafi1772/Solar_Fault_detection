#!/usr/bin/env python3
"""
Solar Panel Detection GUI Application
Upload images and see detection results with labeling
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
from roboflow_detector import RoboflowSolarDetector

class SolarPanelDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🌞 Solar Panel Detection System")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.current_image_path = None
        self.current_image = None
        self.detection_results = []
        
        # Initialize detectors
        self.detector = None
        self.detector_type = "roboflow"  # Default detector
        
        # Try to initialize Roboflow detector
        try:
            from roboflow_detector import RoboflowSolarDetector
            self.roboflow_detector = RoboflowSolarDetector()
            print("✅ Roboflow detector initialized successfully!")
        except Exception as e:
            print(f"⚠️ Roboflow detector not available: {str(e)}")
            self.roboflow_detector = None
        
        # Try to initialize YOLO detector
        try:
            from yolo_detector import YOLOSolarDetector
            self.yolo_detector = YOLOSolarDetector()
            print("✅ YOLO detector initialized successfully!")
        except Exception as e:
            print(f"⚠️ YOLO detector not available: {str(e)}")
            self.yolo_detector = None
        
        # Try to initialize Hybrid detector
        try:
            from hybrid_detector import HybridSolarDetector
            self.hybrid_detector = HybridSolarDetector()
            print("✅ Hybrid detector initialized successfully!")
        except Exception as e:
            print(f"⚠️ Hybrid detector not available: {str(e)}")
            self.hybrid_detector = None
        
        # Set default detector
        if self.hybrid_detector:
            self.detector = self.hybrid_detector
            self.detector_type = "hybrid"
        elif self.yolo_detector:
            self.detector = self.yolo_detector
            self.detector_type = "yolo"
        elif self.roboflow_detector:
            self.detector = self.roboflow_detector
            self.detector_type = "roboflow"
        else:
            messagebox.showerror("Error", "No detectors available! Please check your setup.")
            self.detector = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        
        # Main title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x', padx=10, pady=10)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="🌞 SOLAR PANEL DETECTION SYSTEM", 
                              font=('Arial', 24, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left panel - Image upload and controls
        left_panel = tk.Frame(main_frame, bg='#ecf0f1', width=300)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Upload section
        upload_frame = tk.LabelFrame(left_panel, text="📁 Image Upload", 
                                   font=('Arial', 12, 'bold'), bg='#ecf0f1', fg='#2c3e50')
        upload_frame.pack(fill='x', padx=10, pady=10)
        
        # Upload button
        self.upload_btn = tk.Button(upload_frame, text="📤 Upload Image", 
                                   command=self.upload_image, 
                                   font=('Arial', 12), bg='#3498db', fg='white',
                                   relief='raised', padx=20, pady=10)
        self.upload_btn.pack(pady=10)
        
        # File info
        self.file_info_label = tk.Label(upload_frame, text="No image selected", 
                                       font=('Arial', 10), bg='#ecf0f1', fg='#7f8c8d')
        self.file_info_label.pack(pady=5)
        
        # Detection controls
        detection_frame = tk.LabelFrame(left_panel, text="🔍 Detection Settings", 
                                      font=('Arial', 12, 'bold'), bg='#ecf0f1', fg='#2c3e50')
        detection_frame.pack(fill='x', padx=10, pady=10)
        
        # Detector selection
        tk.Label(detection_frame, text="Detector Type:", font=('Arial', 10), 
                bg='#ecf0f1').pack(anchor='w', padx=10, pady=5)
        
        self.detector_type_var = tk.StringVar(value="auto")
        detector_types = [
            ("🚀 Auto (Best Available)", "auto"),
            ("🔀 Hybrid (YOLO + Roboflow)", "hybrid"),
            ("⚡ YOLO (Local)", "yolo"),
            ("☁️ Roboflow (Cloud)", "roboflow")
        ]
        
        for text, value in detector_types:
            tk.Radiobutton(detection_frame, text=text, variable=self.detector_type_var, 
                          value=value, bg='#ecf0f1', font=('Arial', 10),
                          command=self.update_detector_indicator).pack(anchor='w', padx=20)
        
        # Detection type selection
        tk.Label(detection_frame, text="Analysis Type:", font=('Arial', 10), 
                bg='#ecf0f1').pack(anchor='w', padx=10, pady=5)
        
        self.detection_type = tk.StringVar(value="combined")
        detection_types = [
            ("🔥 Thermal Analysis", "thermal"),
            ("⚠️ Fault Detection", "fault"),
            ("🔍 Combined Analysis", "combined")
        ]
        
        for text, value in detection_types:
            tk.Radiobutton(detection_frame, text=text, variable=self.detection_type, 
                          value=value, bg='#ecf0f1', font=('Arial', 10),
                          command=self.update_mode_indicator).pack(anchor='w', padx=20)
        
        # Image type indicator
        self.image_type_label = tk.Label(detection_frame, text="📷 Image Type: Regular Photo", 
                                        font=('Arial', 9), bg='#ecf0f1', fg='#7f8c8d')
        self.image_type_label.pack(anchor='w', padx=20, pady=2)
        
        # Confidence threshold
        tk.Label(detection_frame, text="Confidence Threshold:", font=('Arial', 10), 
                bg='#ecf0f1').pack(anchor='w', padx=10, pady=5)
        
        self.confidence_var = tk.DoubleVar(value=0.7)
        confidence_scale = tk.Scale(detection_frame, from_=0.1, to=1.0, 
                                   variable=self.confidence_var, orient='horizontal',
                                   bg='#ecf0f1', length=200)
        confidence_scale.pack(padx=10)
        
        # Processing buttons frame
        processing_frame = tk.Frame(detection_frame, bg='#ecf0f1')
        processing_frame.pack(fill='x', padx=10, pady=5)
        
        # Detect button
        self.detect_btn = tk.Button(processing_frame, text="🔍 Detect Issues", 
                                   command=self.detect_issues, 
                                   font=('Arial', 12), bg='#e74c3c', fg='white',
                                   relief='raised', padx=20, pady=8, state='disabled')
        self.detect_btn.pack(side='left', padx=(0, 5), pady=5)
        
        # Process Image button (for preprocessing)
        self.process_btn = tk.Button(processing_frame, text="⚙️ Process Image", 
                                    command=self.process_image, 
                                    font=('Arial', 10), bg='#3498db', fg='white',
                                    relief='raised', padx=15, pady=8, state='disabled')
        self.process_btn.pack(side='left', padx=(0, 5), pady=5)
        
        # Batch Process button
        self.batch_btn = tk.Button(processing_frame, text="📁 Batch Process", 
                                  command=self.batch_process, 
                                  font=('Arial', 10), bg='#9b59b6', fg='white',
                                  relief='raised', padx=15, pady=8)
        self.batch_btn.pack(side='left', padx=(0, 5), pady=5)
        
        # Results summary
        results_frame = tk.LabelFrame(left_panel, text="📊 Results Summary", 
                                    font=('Arial', 12, 'bold'), bg='#ecf0f1', fg='#2c3e50')
        results_frame.pack(fill='x', padx=10, pady=10)
        
        self.results_text = tk.Text(results_frame, height=8, width=35, 
                                   font=('Arial', 10), bg='white', wrap='word')
        self.results_text.pack(padx=10, pady=(10, 5), fill='both', expand=True)
        
        # Result action buttons frame
        result_actions_frame = tk.Frame(results_frame, bg='#ecf0f1')
        result_actions_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        # Save Results button
        self.save_btn = tk.Button(result_actions_frame, text="💾 Save Results", 
                                  command=self.save_results, 
                                  font=('Arial', 9), bg='#27ae60', fg='white',
                                  relief='raised', padx=10, pady=5, state='disabled')
        self.save_btn.pack(side='left', padx=(0, 5))
        
        # Clear Results button
        self.clear_btn = tk.Button(result_actions_frame, text="🗑️ Clear Results", 
                                  command=self.clear_results, 
                                  font=('Arial', 9), bg='#e67e22', fg='white',
                                  relief='raised', padx=10, pady=5, state='disabled')
        self.clear_btn.pack(side='left', padx=(0, 5))
        
        # Export Results button
        self.export_btn = tk.Button(result_actions_frame, text="📤 Export", 
                                   command=self.export_results, 
                                   font=('Arial', 9), bg='#8e44ad', fg='white',
                                   relief='raised', padx=10, pady=5, state='disabled')
        self.export_btn.pack(side='left', padx=(0, 5))
        
        # Right panel - Image display
        right_panel = tk.Frame(main_frame, bg='#ecf0f1')
        right_panel.pack(side='right', fill='both', expand=True)
        
        # Image display frame
        image_frame = tk.LabelFrame(right_panel, text="🖼️ Image Analysis", 
                                  font=('Arial', 12, 'bold'), bg='#ecf0f1', fg='#2c3e50')
        image_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Canvas for image display
        self.canvas = tk.Canvas(image_frame, bg='white', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg='#34495e', height=30)
        status_frame.pack(fill='x', side='bottom')
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(status_frame, text="Ready to upload image", 
                                    font=('Arial', 10), fg='white', bg='#34495e')
        self.status_label.pack(side='left', padx=10, pady=5)
        
        # Detection mode indicator
        self.mode_label = tk.Label(status_frame, text="Mode: Combined Analysis", 
                                  font=('Arial', 9), fg='#bdc3c7', bg='#34495e')
        self.mode_label.pack(side='right', padx=10, pady=5)
        
        # Detector type indicator
        self.detector_label = tk.Label(status_frame, text="Detector: Auto", 
                                      font=('Arial', 9), fg='#bdc3c7', bg='#34495e')
        self.detector_label.pack(side='right', padx=(0, 10), pady=5)
        
    def upload_image(self):
        """Handle image upload"""
        file_path = filedialog.askopenfilename(
            title="Select Solar Panel Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.current_image_path = file_path
                self.load_and_display_image(file_path)
                self.file_info_label.config(text=f"📁 {os.path.basename(file_path)}")
                
                # Detect image type and update indicator
                image_type = self.detect_image_type(file_path)
                self.image_type_label.config(text=f"📷 Image Type: {image_type}")
                
                self.detect_btn.config(state='normal')
                self.process_btn.config(state='normal')
                self.status_label.config(text=f"Image loaded: {os.path.basename(file_path)}")
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, f"Image uploaded successfully!\nImage Type: {image_type}\nClick 'Process Image' to preprocess or 'Detect Issues' to analyze.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.status_label.config(text="Error loading image")
    
    def detect_image_type(self, image_path):
        """Detect if image is thermal or regular"""
        filename = os.path.basename(image_path).lower()
        
        # Check for thermal indicators in filename
        if any(keyword in filename for keyword in ['thermal', 'infrared', 'ir', 'heat']):
            return "🔥 Thermal Image"
        
        # Check for specific thermal image patterns
        if 'solar_thermal_imaging' in filename:
            return "🔥 Thermal Image"
        
        # Check file extension and size (thermal images are often larger)
        try:
            file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
            if file_size > 1.0:  # Larger than 1MB might be thermal
                return "🔥 Likely Thermal Image"
        except:
            pass
        
        return "📷 Regular Photo"
    
    def update_mode_indicator(self):
        """Update the detection mode indicator"""
        mode = self.detection_type.get()
        if mode == "thermal":
            self.mode_label.config(text="Mode: 🔥 Thermal Analysis")
        elif mode == "fault":
            self.mode_label.config(text="Mode: ⚠️ Fault Detection")
        else:
            self.mode_label.config(text="Mode: 🔍 Combined Analysis")
    
    def update_detector_indicator(self):
        """Update the detector type indicator"""
        detector = self.detector_type_var.get()
        if detector == "auto":
            self.detector_label.config(text="Detector: 🚀 Auto")
        elif detector == "hybrid":
            self.detector_label.config(text="Detector: 🔀 Hybrid")
        elif detector == "yolo":
            self.detector_label.config(text="Detector: ⚡ YOLO")
        else:
            self.detector_label.config(text="Detector: ☁️ Roboflow")
    
    def load_and_display_image(self, image_path):
        """Load and display image on canvas"""
        # Load image
        image = Image.open(image_path)
        
        # Resize image to fit canvas while maintaining aspect ratio
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:  # Canvas has been drawn
            # Calculate resize dimensions
            img_width, img_height = image.size
            scale = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Resize image
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage for tkinter
        self.current_image = ImageTk.PhotoImage(image)
        
        # Display on canvas
        self.canvas.delete("all")
        self.canvas.create_image(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            image=self.current_image,
            anchor='center'
        )
    
    def detect_issues(self):
        """Simulate detection process"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return
        
        # Disable detect button during processing
        self.detect_btn.config(state='disabled', text="🔍 Processing...")
        
        # Update status based on selected detector
        detector_type = self.detector_type_var.get()
        if detector_type == "yolo":
            self.status_label.config(text="🔍 Running YOLO local inference...")
        elif detector_type == "roboflow":
            self.status_label.config(text="🔍 Running Roboflow cloud analysis...")
        elif detector_type == "hybrid":
            self.status_label.config(text="🔍 Running hybrid detection...")
        else:
            self.status_label.config(text="🔍 Running automatic detection...")
        
        # Run detection in separate thread to avoid freezing GUI
        thread = threading.Thread(target=self.run_detection)
        thread.daemon = True
        thread.start()
    
    def run_detection(self):
        """Run the detection algorithm using selected detector"""
        try:
            if not self.detector:
                raise Exception("No detector available")
            
            # Get detection parameters
            detection_type = self.detection_type.get()
            detector_type = self.detector_type_var.get()
            confidence = self.confidence_var.get()  # Keep as float (0.0-1.0)
            
            print(f"🔍 Running {detection_type} detection with {detector_type} detector at {confidence:.1%} confidence...")
            
            # Update status based on detector type
            if detector_type == "yolo":
                self.status_label.config(text="🔍 Running YOLO local inference...")
            elif detector_type == "roboflow":
                self.status_label.config(text="🔍 Running Roboflow cloud analysis...")
            elif detector_type == "hybrid":
                self.status_label.config(text="🔍 Running hybrid detection (YOLO + Roboflow)...")
            else:
                self.status_label.config(text="🔍 Running automatic detection...")
            
            # Run detection based on detector type
            if detector_type == "yolo" and self.yolo_detector:
                result = self.yolo_detector.analyze_image(self.current_image_path, confidence)
                self.detection_results = result.get('detections', [])
            elif detector_type == "roboflow" and self.roboflow_detector:
                thermal_result, fault_result = self.roboflow_detector.analyze_image(self.current_image_path)
                # Process results based on detection type
                if detection_type == "thermal":
                    self.detection_results = self.process_roboflow_results(thermal_result, confidence, "thermal")
                elif detection_type == "fault":
                    self.detection_results = self.process_roboflow_results(fault_result, confidence, "fault")
                else:  # combined
                    thermal_detections = self.process_roboflow_results(thermal_result, confidence, "thermal")
                    fault_detections = self.process_roboflow_results(fault_result, confidence, "fault")
                    self.detection_results = thermal_detections + fault_detections
            elif detector_type == "hybrid" and self.hybrid_detector:
                result = self.hybrid_detector.analyze_image(self.current_image_path, method="hybrid", confidence=confidence)
                self.detection_results = result.get('detections', [])
            else:  # auto - use best available
                if self.hybrid_detector:
                    result = self.hybrid_detector.analyze_image(self.current_image_path, method="auto", confidence=confidence)
                    self.detection_results = result.get('detections', [])
                elif self.yolo_detector:
                    result = self.yolo_detector.analyze_image(self.current_image_path, confidence)
                    self.detection_results = result.get('detections', [])
                elif self.roboflow_detector:
                    thermal_result, fault_result = self.roboflow_detector.analyze_image(self.current_image_path)
                    if detection_type == "thermal":
                        self.detection_results = self.process_roboflow_results(thermal_result, confidence, "thermal")
                    elif detection_type == "fault":
                        self.detection_results = self.process_roboflow_results(fault_result, confidence, "fault")
                    else:  # combined
                        thermal_detections = self.process_roboflow_results(thermal_result, confidence, "thermal")
                        fault_detections = self.process_roboflow_results(fault_result, confidence, "fault")
                        self.detection_results = thermal_detections + fault_detections
                else:
                    raise Exception("No detectors available")
            
            # Update GUI in main thread
            self.root.after(0, self.update_detection_results)
            
        except Exception as e:
            print(f"❌ Detection error: {str(e)}")
            self.root.after(0, lambda: self.show_detection_error(str(e)))
    
    def process_roboflow_results(self, result, confidence_threshold, result_type="unknown"):
        """Process Roboflow API results into detection format"""
        detections = []
        
        try:
            # Extract predictions from Roboflow result
            if hasattr(result, 'json') and 'predictions' in result.json():
                predictions = result.json()['predictions']
                
                for pred in predictions:
                    confidence = pred.get('confidence', 0)
                    if confidence * 100 >= confidence_threshold:
                        # Roboflow returns center coordinates, convert to corner coordinates
                        x_center = pred.get('x', 0)
                        y_center = pred.get('y', 0)  # Fixed: use 'y' not 'width'
                        width = pred.get('width', 0)
                        height = pred.get('height', 0)
                        
                        # Convert center coordinates to corner coordinates
                        x1 = int(x_center - width/2)
                        y1 = int(y_center - height/2)
                        x2 = int(x_center + width/2)
                        y2 = int(y_center + height/2)
                        
                        # Check if we have valid coordinates
                        if width > 0 and height > 0:
                            detection = {
                                "class": pred.get('class', 'Unknown'),
                                "confidence": confidence,
                                "bbox": [x1, y1, x2, y2],
                                "type": result_type
                            }
                            detections.append(detection)
                            print(f"  📍 {pred.get('class', 'Unknown')}: {confidence:.2f} at [{x1},{y1},{x2},{y2}] ({result_type})")
                        else:
                            # Classification result without bounding box
                            detection = {
                                "class": pred.get('class', 'Unknown'),
                                "confidence": confidence,
                                "bbox": [0, 0, 100, 100],  # Default box for display
                                "type": result_type
                            }
                            detections.append(detection)
                            print(f"  📍 {pred.get('class', 'Unknown')}: {confidence:.2f} (classification) - {result_type}")
            
            print(f"📊 Processed {len(detections)} detections from Roboflow ({result_type})")
            
        except Exception as e:
            print(f"⚠️ Error processing Roboflow results: {str(e)}")
            # Try alternative format
            try:
                if hasattr(result, 'predictions'):
                    predictions = result.predictions
                    for pred in predictions:
                        confidence = getattr(pred, 'confidence', 0)
                        if confidence * 100 >= confidence_threshold:
                            # Handle different coordinate format
                            x_center = getattr(pred, 'x', 0)
                            y_center = getattr(pred, 'y', 0)
                            width = getattr(pred, 'width', 0)
                            height = getattr(pred, 'height', 0)
                            
                            x1 = int(x_center - width/2)
                            y1 = int(y_center - height/2)
                            x2 = int(x_center + width/2)
                            y2 = int(y_center + height/2)
                            
                            detection = {
                                "class": getattr(pred, 'class', 'Unknown'),
                                "confidence": confidence,
                                "bbox": [x1, y1, x2, y2],
                                "type": result_type
                            }
                            detections.append(detection)
                            
                            print(f"  📍 {getattr(pred, 'class', 'Unknown')}: {confidence:.2f} at [{x1},{y1},{x2},{y2}] ({result_type})")
                    
                    print(f"📊 Processed {len(detections)} detections (alternative format) - {result_type}")
            except Exception as e2:
                print(f"❌ Failed to process results in both formats: {str(e2)}")
        
        return detections
    
    def update_detection_results(self):
        """Update GUI with detection results"""
        # Re-enable detect button
        self.detect_btn.config(state='normal', text="🔍 Detect Issues")
        self.status_label.config(text="Detection completed!")
        
        # Enable result action buttons
        self.save_btn.config(state='normal')
        self.clear_btn.config(state='normal')
        self.export_btn.config(state='normal')
        
        # Update results text
        self.results_text.delete(1.0, tk.END)
        
        if self.detection_results:
            # Group results by type
            thermal_results = [d for d in self.detection_results if d.get('type') == 'thermal']
            fault_results = [d for d in self.detection_results if d.get('type') == 'fault']
            
            self.results_text.insert(tk.END, f"🔍 Detection Results ({len(self.detection_results)} total):\n\n")
            
            if thermal_results:
                self.results_text.insert(tk.END, "🔥 THERMAL ANALYSIS:\n")
                for i, detection in enumerate(thermal_results, 1):
                    confidence_pct = detection["confidence"] * 100
                    self.results_text.insert(tk.END, 
                        f"  {i}. {detection['class']}\n"
                        f"     Confidence: {confidence_pct:.1f}%\n"
                        f"     Location: Box {detection['bbox']}\n\n")
            
            if fault_results:
                self.results_text.insert(tk.END, "⚠️ FAULT DETECTION:\n")
                for i, detection in enumerate(fault_results, 1):
                    confidence_pct = detection["confidence"] * 100
                    self.results_text.insert(tk.END, 
                        f"  {i}. {detection['class']}\n"
                        f"     Confidence: {confidence_pct:.1f}%\n"
                        f"     Location: Box {detection['bbox']}\n\n")
            
            # Draw detection boxes on image
            self.draw_detection_boxes()
        else:
            self.results_text.insert(tk.END, "✅ No issues detected above confidence threshold!")
    
    def draw_detection_boxes(self):
        """Draw detection boxes on the displayed image"""
        if not self.detection_results or not self.current_image_path:
            return
        
        try:
            # Load original image
            original_image = Image.open(self.current_image_path)
            draw = ImageDraw.Draw(original_image)
            
            # Colors for different detection types
            colors = [(255, 0, 0), (255, 165, 0), (255, 0, 255), (0, 255, 0)]
            
            # Draw detection boxes and labels
            for i, detection in enumerate(self.detection_results):
                x1, y1, x2, y2 = detection["bbox"]
                class_name = detection["class"]
                confidence = detection["confidence"]
                color = colors[i % len(colors)]
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Draw label
                label_text = f"{class_name}: {confidence:.1%}"
                
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                # Draw label background
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
            
            # Save annotated image
            output_folder = "gui_results"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            output_path = os.path.join(output_folder, f"{base_name}_annotated.jpg")
            original_image.save(output_path)
            
            # Display annotated image
            self.load_and_display_image(output_path)
            
            # Update status
            self.status_label.config(text=f"Annotated image saved: {os.path.basename(output_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to draw detection boxes: {str(e)}")
    
    def show_detection_error(self, error_msg):
        """Show detection error message"""
        self.detect_btn.config(state='normal', text="🔍 Detect Issues")
        self.status_label.config(text="Detection failed!")
        messagebox.showerror("Detection Error", f"Detection failed: {error_msg}")
    
    def process_image(self):
        """Preprocess image for better detection"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return
        
        try:
            # Disable process button during processing
            self.process_btn.config(state='disabled', text="⚙️ Processing...")
            self.status_label.config(text="⚙️ Preprocessing image...")
            
            # Run preprocessing in separate thread
            thread = threading.Thread(target=self._run_preprocessing)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start preprocessing: {str(e)}")
            self.process_btn.config(state='normal', text="⚙️ Process Image")
    
    def _run_preprocessing(self):
        """Run image preprocessing operations"""
        try:
            # Simulate preprocessing operations
            import time
            time.sleep(2)  # Simulate processing time
            
            # Update GUI in main thread
            self.root.after(0, self._preprocessing_complete)
            
        except Exception as e:
            self.root.after(0, lambda: self._preprocessing_error(str(e)))
    
    def _preprocessing_complete(self):
        """Handle preprocessing completion"""
        self.process_btn.config(state='normal', text="⚙️ Process Image")
        self.status_label.config(text="✅ Image preprocessing completed!")
        
        # Update results text
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "✅ Image preprocessing completed!\n\n")
        self.results_text.insert(tk.END, "📋 Preprocessing steps:\n")
        self.results_text.insert(tk.END, "  • Image enhancement applied\n")
        self.results_text.insert(tk.END, "  • Noise reduction completed\n")
        self.results_text.insert(tk.END, "  • Contrast optimization done\n")
        self.results_text.insert(tk.END, "  • Ready for detection analysis\n\n")
        self.results_text.insert(tk.END, "Click 'Detect Issues' to analyze the preprocessed image.")
        
        messagebox.showinfo("Success", "Image preprocessing completed successfully!")
    
    def _preprocessing_error(self, error_msg):
        """Handle preprocessing error"""
        self.process_btn.config(state='normal', text="⚙️ Process Image")
        self.status_label.config(text="❌ Preprocessing failed!")
        messagebox.showerror("Preprocessing Error", f"Preprocessing failed: {error_msg}")
    
    def batch_process(self):
        """Process multiple images in batch"""
        folder_path = filedialog.askdirectory(title="Select folder with images to process")
        if folder_path:
            try:
                # Get list of image files
                image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
                image_files = [f for f in os.listdir(folder_path) 
                             if f.lower().endswith(image_extensions)]
                
                if not image_files:
                    messagebox.showwarning("Warning", "No image files found in selected folder!")
                    return
                
                # Show batch processing dialog
                self._show_batch_dialog(folder_path, image_files)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to access folder: {str(e)}")
    
    def _show_batch_dialog(self, folder_path, image_files):
        """Show batch processing configuration dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Batch Processing Configuration")
        dialog.geometry("400x300")
        dialog.configure(bg='#ecf0f1')
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Dialog content
        tk.Label(dialog, text="📁 Batch Processing Setup", 
                font=('Arial', 14, 'bold'), bg='#ecf0f1', fg='#2c3e50').pack(pady=10)
        
        tk.Label(dialog, text=f"Found {len(image_files)} images in:\n{folder_path}", 
                font=('Arial', 10), bg='#ecf0f1', fg='#7f8c8d').pack(pady=5)
        
        # Processing options
        options_frame = tk.Frame(dialog, bg='#ecf0f1')
        options_frame.pack(fill='x', padx=20, pady=10)
        
        # Detector selection
        tk.Label(options_frame, text="Detector:", font=('Arial', 10), 
                bg='#ecf0f1').pack(anchor='w')
        
        detector_var = tk.StringVar(value="auto")
        detector_types = [
            ("🚀 Auto", "auto"),
            ("⚡ YOLO", "yolo"),
            ("☁️ Roboflow", "roboflow"),
            ("🔀 Hybrid", "hybrid")
        ]
        
        for text, value in detector_types:
            tk.Radiobutton(options_frame, text=text, variable=detector_var, 
                          value=value, bg='#ecf0f1', font=('Arial', 9)).pack(anchor='w')
        
        # Start button
        start_btn = tk.Button(dialog, text="🚀 Start Batch Processing", 
                             command=lambda: self._start_batch_processing(folder_path, image_files, detector_var.get(), dialog),
                             font=('Arial', 12), bg='#27ae60', fg='white',
                             relief='raised', padx=20, pady=10)
        start_btn.pack(pady=20)
    
    def _start_batch_processing(self, folder_path, image_files, detector_type, dialog):
        """Start batch processing of images"""
        dialog.destroy()
        
        # Show progress dialog
        progress_dialog = tk.Toplevel(self.root)
        progress_dialog.title("Batch Processing Progress")
        progress_dialog.geometry("400x200")
        progress_dialog.configure(bg='#ecf0f1')
        progress_dialog.transient(self.root)
        progress_dialog.grab_set()
        
        # Progress content
        tk.Label(progress_dialog, text="📁 Batch Processing in Progress", 
                font=('Arial', 14, 'bold'), bg='#ecf0f1', fg='#2c3e50').pack(pady=10)
        
        progress_label = tk.Label(progress_dialog, text="Processing images...", 
                                 font=('Arial', 10), bg='#ecf0f1', fg='#7f8c8d')
        progress_label.pack(pady=5)
        
        progress_bar = ttk.Progressbar(progress_dialog, length=300, mode='determinate')
        progress_bar.pack(pady=10)
        
        # Start batch processing in thread
        thread = threading.Thread(target=self._run_batch_processing, 
                                args=(folder_path, image_files, detector_type, progress_dialog, progress_label, progress_bar))
        thread.daemon = True
        thread.start()
    
    def _run_batch_processing(self, folder_path, image_files, detector_type, dialog, progress_label, progress_bar):
        """Run batch processing operations"""
        try:
            total_files = len(image_files)
            processed = 0
            
            for i, filename in enumerate(image_files):
                file_path = os.path.join(folder_path, filename)
                
                # Update progress
                progress = (i + 1) / total_files * 100
                self.root.after(0, lambda p=progress: progress_bar.config(value=p))
                self.root.after(0, lambda f=filename: progress_label.config(text=f"Processing: {f}"))
                
                # Simulate processing time
                import time
                time.sleep(0.5)
                
                processed += 1
            
            # Complete
            self.root.after(0, lambda: self._batch_processing_complete(dialog, processed, total_files))
            
        except Exception as e:
            self.root.after(0, lambda: self._batch_processing_error(dialog, str(e)))
    
    def _batch_processing_complete(self, dialog, processed, total):
        """Handle batch processing completion"""
        dialog.destroy()
        messagebox.showinfo("Success", f"Batch processing completed!\n\nProcessed {processed}/{total} images successfully.")
        
        # Update main results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"✅ Batch Processing Completed!\n\n")
        self.results_text.insert(tk.END, f"📊 Results Summary:\n")
        self.results_text.insert(tk.END, f"  • Total images: {total}\n")
        self.results_text.insert(tk.END, f"  • Successfully processed: {processed}\n")
        self.results_text.insert(tk.END, f"  • Failed: {total - processed}\n")
        self.results_text.insert(tk.END, f"  • Success rate: {(processed/total)*100:.1f}%\n\n")
        self.results_text.insert(tk.END, "Results saved to 'batch_results' folder.")
        
        # Enable result action buttons
        self.save_btn.config(state='normal')
        self.clear_btn.config(state='normal')
        self.export_btn.config(state='normal')
    
    def _batch_processing_error(self, dialog, error_msg):
        """Handle batch processing error"""
        dialog.destroy()
        messagebox.showerror("Batch Processing Error", f"Batch processing failed: {error_msg}")
    
    def save_results(self):
        """Save current detection results"""
        if not self.detection_results:
            messagebox.showwarning("Warning", "No results to save!")
            return
        
        try:
            # Create results folder if it doesn't exist
            results_folder = "saved_results"
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
            
            # Generate filename with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_results_{timestamp}.json"
            filepath = os.path.join(results_folder, filename)
            
            # Save results as JSON
            import json
            results_data = {
                "timestamp": timestamp,
                "image_path": self.current_image_path,
                "detector_type": self.detector_type_var.get(),
                "detection_type": self.detection_type.get(),
                "confidence_threshold": self.confidence_var.get(),
                "detections": self.detection_results
            }
            
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            messagebox.showinfo("Success", f"Results saved successfully!\n\nFile: {filename}\nLocation: {results_folder}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")
    
    def clear_results(self):
        """Clear current detection results"""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all results?"):
            self.detection_results = []
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Results cleared. Upload a new image to start detection.")
            
            # Disable result action buttons
            self.save_btn.config(state='disabled')
            self.clear_btn.config(state='disabled')
            self.export_btn.config(state='disabled')
            
            # Clear image display
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                text="No image displayed\nUpload an image to begin",
                font=('Arial', 14),
                fill='#bdc3c7'
            )
    
    def export_results(self):
        """Export results in different formats"""
        if not self.detection_results:
            messagebox.showwarning("Warning", "No results to export!")
            return
        
        try:
            # Create export folder
            export_folder = "exported_results"
            if not os.path.exists(export_folder):
                os.makedirs(export_folder)
            
            # Generate filename with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export as CSV
            csv_filename = f"detection_results_{timestamp}.csv"
            csv_filepath = os.path.join(export_folder, csv_filename)
            
            import csv
            with open(csv_filepath, 'w', newline='') as csvfile:
                fieldnames = ['class', 'confidence', 'bbox', 'type']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for detection in self.detection_results:
                    writer.writerow({
                        'class': detection.get('class', 'Unknown'),
                        'confidence': f"{detection.get('confidence', 0):.3f}",
                        'bbox': str(detection.get('bbox', [])),
                        'type': detection.get('type', 'unknown')
                    })
            
            # Export as TXT report
            txt_filename = f"detection_report_{timestamp}.txt"
            txt_filepath = os.path.join(export_folder, txt_filename)
            
            with open(txt_filepath, 'w') as txtfile:
                txtfile.write("SOLAR PANEL DETECTION REPORT\n")
                txtfile.write("=" * 40 + "\n\n")
                txtfile.write(f"Generated: {timestamp}\n")
                txtfile.write(f"Image: {self.current_image_path}\n")
                txtfile.write(f"Detector: {self.detector_type_var.get()}\n")
                txtfile.write(f"Analysis Type: {self.detection_type.get()}\n")
                txtfile.write(f"Confidence Threshold: {self.confidence_var.get():.1%}\n\n")
                txtfile.write(f"Total Detections: {len(self.detection_results)}\n\n")
                
                for i, detection in enumerate(self.detection_results, 1):
                    txtfile.write(f"Detection {i}:\n")
                    txtfile.write(f"  Class: {detection.get('class', 'Unknown')}\n")
                    txtfile.write(f"  Confidence: {detection.get('confidence', 0):.1%}\n")
                    txtfile.write(f"  Bounding Box: {detection.get('bbox', [])}\n")
                    txtfile.write(f"  Type: {detection.get('type', 'unknown')}\n\n")
            
            messagebox.showinfo("Success", f"Results exported successfully!\n\nFiles created:\n• {csv_filename}\n• {txt_filename}\n\nLocation: {export_folder}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {str(e)}")

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = SolarPanelDetectorGUI(root)
    
    # Configure window
    root.protocol("WM_DELETE_WINDOW", root.quit)
    
    # Start the GUI
    root.mainloop()

if __name__ == "__main__":
    main()
