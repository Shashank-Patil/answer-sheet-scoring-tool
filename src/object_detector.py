"""
Object detection utilities using YOLO for answer sheet regions.
"""

import os
import shutil
from typing import List, Dict
from ultralytics import YOLO
import cv2


class ObjectDetector:
    """Handles YOLO object detection for answer sheet regions."""
    
    def __init__(self, model_path: str):
        """
        Initialize object detector with YOLO model.
        
        Args:
            model_path: Path to the trained YOLO model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_path = model_path
        self.model = None
    
    def load_model(self):
        """Load the YOLO model."""
        if self.model is None:
            self.model = YOLO(self.model_path)
            print(f"Loaded YOLO model from {self.model_path}")
    
    def detect_objects(self, source_dir: str, output_dir: str, conf: float = 0.25, 
                      save_txt: bool = True, save_crops: bool = True) -> str:
        """
        Run YOLO detection on images in source directory.
        
        Args:
            source_dir: Directory containing input images
            output_dir: Base directory for detection outputs
            conf: Confidence threshold for detections
            save_txt: Whether to save detection labels as text files
            save_crops: Whether to save cropped detected regions
            
        Returns:
            Path to the detection results directory
        """
        self.load_model()
        
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Run YOLO detection
        results = self.model.predict(
            source=source_dir,
            conf=conf,
            save_txt=save_txt,
            save=True,
            save_crop=save_crops,
            project=output_dir,
            name='detect'
        )
        
        # Return path to results
        results_dir = os.path.join(output_dir, 'detect')
        print(f"Detection results saved to {results_dir}")
        return results_dir
    
    def extract_crops(self, detection_results_dir: str, output_dir: str) -> Dict[str, List[str]]:
        """
        Extract and organize cropped detection regions.
        
        Args:
            detection_results_dir: Directory containing YOLO detection results
            output_dir: Directory to save organized crops
            
        Returns:
            Dictionary mapping class names to lists of crop image paths
        """
        crops_dir = os.path.join(detection_results_dir, 'crops')
        
        if not os.path.exists(crops_dir):
            print(f"No crops found in {crops_dir}")
            return {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        extracted_crops = {}
        
        # Iterate through class subfolders
        for class_folder in os.listdir(crops_dir):
            class_path = os.path.join(crops_dir, class_folder)
            
            if os.path.isdir(class_path):
                class_output_dir = os.path.join(output_dir, class_folder)
                os.makedirs(class_output_dir, exist_ok=True)
                
                crop_paths = []
                
                # Copy all images from class folder
                for file in os.listdir(class_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        src_path = os.path.join(class_path, file)
                        dst_path = os.path.join(class_output_dir, file)
                        shutil.copy2(src_path, dst_path)
                        crop_paths.append(dst_path)
                
                extracted_crops[class_folder] = crop_paths
                print(f"Extracted {len(crop_paths)} crops for class {class_folder}")
        
        return extracted_crops
    
    def process_images(self, source_dir: str, base_output_dir: str, 
                      conf: float = 0.25) -> Dict[str, List[str]]:
        """
        Complete pipeline: detect objects and extract crops.
        
        Args:
            source_dir: Directory containing input images
            base_output_dir: Base directory for all outputs
            conf: Confidence threshold for detections
            
        Returns:
            Dictionary mapping class names to lists of crop image paths
        """
        # Run detection
        detection_dir = os.path.join(base_output_dir, 'detection_results')
        results_dir = self.detect_objects(source_dir, detection_dir, conf)
        
        # Extract crops
        crops_dir = os.path.join(base_output_dir, 'extracted_crops')
        extracted_crops = self.extract_crops(results_dir, crops_dir)
        
        return extracted_crops