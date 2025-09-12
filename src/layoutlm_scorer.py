"""
Similarity scoring using LayoutLMv3 model for document understanding.
Better suited for answer sheet comparison than CLIP.
"""

import os
import torch
import numpy as np
from PIL import Image
try:
    from transformers import LayoutLMv3Processor, LayoutLMv3Model
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers library not available. Using fallback similarity calculation.")
    TRANSFORMERS_AVAILABLE = False
    LayoutLMv3Processor = None
    LayoutLMv3Model = None
from typing import List, Dict, Tuple
import cv2
try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    # Fallback implementation
    def cosine_similarity(X, Y):
        import numpy as np
        X = np.array(X)
        Y = np.array(Y)
        return np.dot(X, Y.T) / (np.linalg.norm(X) * np.linalg.norm(Y))


class LayoutLMv3Scorer:
    """Handles LayoutLMv3-based similarity scoring between answer images."""
    
    def __init__(self, device: str = None):
        """
        Initialize similarity scorer with LayoutLMv3 model.
        
        Args:
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = None
        self.processor = None
    
    def load_model(self):
        """Load the LayoutLMv3 model and processor."""
        if self.model is None:
            try:
                # Use Microsoft's LayoutLMv3 base model
                model_name = "microsoft/layoutlmv3-base"
                
                self.processor = LayoutLMv3Processor.from_pretrained(model_name)
                self.model = LayoutLMv3Model.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
                
                print(f"Loaded LayoutLMv3 model on device: {self.device}")
            except Exception as e:
                print(f"Error loading LayoutLMv3 model: {e}")
                print("Falling back to a simpler visual similarity approach...")
                self.model = "fallback"
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract features from an image using LayoutLMv3.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Feature vector as numpy array
        """
        if self.model == "fallback":
            return self._extract_visual_features_fallback(image_path)
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # For LayoutLMv3, we need text and bounding boxes
            # Since we don't have OCR data, we'll use empty text and boxes
            # This will make the model focus on visual features
            text = ""
            boxes = []
            
            # Process the inputs
            inputs = self.processor(
                image, 
                text=text, 
                boxes=boxes, 
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use the pooled output as the feature representation
                features = outputs.last_hidden_state.mean(dim=1)  # Average pooling
                features = features.cpu().numpy().flatten()
            
            return features
            
        except Exception as e:
            print(f"Error extracting features with LayoutLMv3: {e}")
            return self._extract_visual_features_fallback(image_path)
    
    def _extract_visual_features_fallback(self, image_path: str) -> np.ndarray:
        """
        Fallback method using traditional computer vision features.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Feature vector as numpy array
        """
        try:
            # Load image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Resize to standard size
            image = cv2.resize(image, (224, 224))
            
            # Extract multiple types of features
            features = []
            
            # 1. Histogram features
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            features.extend(hist.flatten())
            
            # 2. Edge features using Canny
            edges = cv2.Canny(image, 50, 150)
            edge_hist = cv2.calcHist([edges], [0], None, [256], [0, 256])
            features.extend(edge_hist.flatten())
            
            # 3. Texture features using LBP-like approach
            # Simple local binary pattern approximation
            kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
            texture = cv2.filter2D(image, -1, kernel)
            texture_hist = cv2.calcHist([texture], [0], None, [256], [0, 256])
            features.extend(texture_hist.flatten())
            
            # 4. Structural features - divide image into regions
            h, w = image.shape
            regions = [
                image[0:h//2, 0:w//2],      # Top-left
                image[0:h//2, w//2:w],      # Top-right
                image[h//2:h, 0:w//2],      # Bottom-left
                image[h//2:h, w//2:w]       # Bottom-right
            ]
            
            for region in regions:
                region_mean = np.mean(region)
                region_std = np.std(region)
                features.extend([region_mean, region_std])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error in fallback feature extraction: {e}")
            # Return random features as last resort
            return np.random.random(1024).astype(np.float32)
    
    def calculate_similarity_score(self, image1_path: str, image2_path: str) -> float:
        """
        Calculate similarity score between two images using LayoutLMv3 or fallback.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            
        Returns:
            Similarity score between 0 and 1
        """
        self.load_model()
        
        # Extract features from both images
        features1 = self.extract_features(image1_path)
        features2 = self.extract_features(image2_path)
        
        # Calculate cosine similarity
        similarity = cosine_similarity([features1], [features2])[0, 0]
        
        # Ensure similarity is between 0 and 1
        similarity = max(0, min(1, (similarity + 1) / 2))  # Normalize from [-1,1] to [0,1]
        
        return float(similarity)
    
    def find_best_match(self, reference_image: str, candidate_images: List[str]) -> Tuple[str, float, int]:
        """
        Find the best matching image from candidates for a reference image.
        
        Args:
            reference_image: Path to reference image
            candidate_images: List of candidate image paths
            
        Returns:
            Tuple of (best_match_path, similarity_score, index)
        """
        if not candidate_images:
            return None, 0.0, -1
        
        similarity_scores = []
        
        for candidate_image in candidate_images:
            score = self.calculate_similarity_score(reference_image, candidate_image)
            similarity_scores.append(score)
        
        # Find maximum similarity
        max_index = max(enumerate(similarity_scores), key=lambda x: x[1])[0]
        max_score = similarity_scores[max_index]
        best_match_path = candidate_images[max_index]
        
        return best_match_path, max_score, max_index
    
    def compare_image_sets(self, reference_dir: str, student_dir: str, 
                          manual_scores: List[float] = None,
                          display_matches: bool = False,
                          similarity_threshold: float = 0.85) -> Dict:
        """
        Compare all images in reference directory with student directory.
        
        Args:
            reference_dir: Directory containing reference answer images
            student_dir: Directory containing student answer images
            manual_scores: List of manual scoring weights for each question
            display_matches: Whether to display matched image pairs
            similarity_threshold: Minimum similarity score to award full marks (default: 0.85)
            
        Returns:
            Dictionary containing comparison results
        """
        self.load_model()
        
        if not os.path.exists(reference_dir) or not os.path.exists(student_dir):
            raise FileNotFoundError("Reference or student directory not found")
        
        # Get image files
        reference_images = [f for f in os.listdir(reference_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        student_images = [f for f in os.listdir(student_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not reference_images or not student_images:
            print("No images found in one or both directories")
            return {}
        
        # Default manual scores if not provided
        if manual_scores is None:
            manual_scores = [3.0] * len(reference_images)
        elif len(manual_scores) < len(reference_images):
            # Extend manual_scores if too short
            manual_scores.extend([3.0] * (len(reference_images) - len(manual_scores)))
        
        results = {
            'individual_scores': [],
            'similarity_scores': [],
            'total_score': 0.0,
            'matches': []
        }
        
        student_image_paths = [os.path.join(student_dir, img) for img in student_images]
        
        for i, ref_image in enumerate(reference_images):
            ref_path = os.path.join(reference_dir, ref_image)
            
            # Find best match
            best_match_path, max_similarity, match_index = self.find_best_match(
                ref_path, student_image_paths
            )
            
            if best_match_path:
                # Apply threshold-based scoring: full marks if >= threshold, 0 if below
                if max_similarity >= similarity_threshold:
                    weighted_score = manual_scores[i]  # Full marks
                    score_status = "PASS"
                else:
                    weighted_score = 0.0  # No marks
                    score_status = "FAIL"
                
                results['individual_scores'].append(weighted_score)
                results['similarity_scores'].append(max_similarity)
                results['matches'].append({
                    'reference': ref_path,
                    'student': best_match_path,
                    'similarity': max_similarity,
                    'weighted_score': weighted_score,
                    'passed_threshold': max_similarity >= similarity_threshold
                })
                
                print(f"Image: {ref_image}, Similarity: {max_similarity:.4f} ({score_status}), "
                      f"Score: {weighted_score:.2f}/{manual_scores[i]:.2f}")
                
                # Display matches if requested
                if display_matches:
                    self._display_match(ref_path, best_match_path)
            else:
                results['individual_scores'].append(0.0)
                results['similarity_scores'].append(0.0)
                print(f"No match found for reference image: {ref_image}")
        
        results['total_score'] = sum(results['individual_scores'])
        
        print(f"\nIndividual Scores: {results['individual_scores']}")
        print(f"Total Score: {results['total_score']:.2f}")
        
        return results
    
    def _display_match(self, ref_path: str, student_path: str):
        """
        Display matched image pairs using OpenCV.
        
        Args:
            ref_path: Path to reference image
            student_path: Path to student image
        """
        try:
            ref_image = cv2.imread(ref_path)
            student_image = cv2.imread(student_path)
            
            if ref_image is not None and student_image is not None:
                # In a Jupyter environment, you would use cv2_imshow
                # For now, we'll just print the paths
                print(f"Reference: {ref_path}")
                print(f"Student: {student_path}")
                print("---")
        except Exception as e:
            print(f"Error displaying images: {e}")
    
    def batch_compare_classes(self, reference_base_dir: str, student_base_dir: str,
                             class_names: List[str], manual_scores: List[float] = None,
                             similarity_threshold: float = 0.85) -> Dict:
        """
        Compare multiple classes of detected objects.
        
        Args:
            reference_base_dir: Base directory containing reference class folders
            student_base_dir: Base directory containing student class folders
            class_names: List of class names to compare
            manual_scores: Manual scoring weights
            similarity_threshold: Minimum similarity score to award full marks
            
        Returns:
            Dictionary containing results for all classes
        """
        all_results = {}
        
        for class_name in class_names:
            ref_class_dir = os.path.join(reference_base_dir, class_name)
            student_class_dir = os.path.join(student_base_dir, class_name)
            
            if os.path.exists(ref_class_dir) and os.path.exists(student_class_dir):
                print(f"\nComparing class: {class_name}")
                results = self.compare_image_sets(
                    ref_class_dir, student_class_dir, manual_scores, 
                    similarity_threshold=similarity_threshold
                )
                all_results[class_name] = results
            else:
                print(f"Skipping class {class_name} - directories not found")
                all_results[class_name] = {'total_score': 0.0, 'individual_scores': []}
        
        return all_results