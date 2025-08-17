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
        
        # Initialize Roboflow detector
        try:
            self.detector = RoboflowSolarDetector()
            print("✅ Roboflow detector initialized successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize Roboflow detector: {str(e)}")
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
        
        # Detection type selection
        tk.Label(detection_frame, text="Detection Type:", font=('Arial', 10), 
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
        
        # Detect button
        self.detect_btn = tk.Button(detection_frame, text="🔍 Detect Issues", 
                                   command=self.detect_issues, 
                                   font=('Arial', 12), bg='#e74c3c', fg='white',
                                   relief='raised', padx=20, pady=10, state='disabled')
        self.detect_btn.pack(pady=10)
        
        # Results summary
        results_frame = tk.LabelFrame(left_panel, text="📊 Results Summary", 
                                    font=('Arial', 12, 'bold'), bg='#ecf0f1', fg='#2c3e50')
        results_frame.pack(fill='x', padx=10, pady=10)
        
        self.results_text = tk.Text(results_frame, height=8, width=35, 
                                   font=('Arial', 10), bg='white', wrap='word')
        self.results_text.pack(padx=10, pady=10, fill='both', expand=True)
        
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
                self.status_label.config(text=f"Image loaded: {os.path.basename(file_path)}")
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, f"Image uploaded successfully!\nImage Type: {image_type}\nClick 'Detect Issues' to analyze.")
                
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
        self.status_label.config(text="🔍 Running Roboflow API analysis...")
        
        # Run detection in separate thread to avoid freezing GUI
        thread = threading.Thread(target=self.run_detection)
        thread.daemon = True
        thread.start()
    
    def run_detection(self):
        """Run the detection algorithm using real Roboflow API"""
        try:
            if not self.detector:
                raise Exception("Roboflow detector not initialized")
            
            # Get detection type and confidence
            detection_type = self.detection_type.get()
            confidence = int(self.confidence_var.get() * 100)  # Convert to percentage
            
            print(f"🔍 Running {detection_type} detection with {confidence}% confidence...")
            
            # Run real Roboflow detection
            thermal_result, fault_result = self.detector.analyze_image(self.current_image_path)
            
            # Process results based on detection type
            if detection_type == "thermal":
                self.detection_results = self.process_roboflow_results(thermal_result, confidence, "thermal")
            elif detection_type == "fault":
                self.detection_results = self.process_roboflow_results(fault_result, confidence, "fault")
            else:  # combined
                thermal_detections = self.process_roboflow_results(thermal_result, confidence, "thermal")
                fault_detections = self.process_roboflow_results(fault_result, confidence, "fault")
                self.detection_results = thermal_detections + fault_detections
            
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
