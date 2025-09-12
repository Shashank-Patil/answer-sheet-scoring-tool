"""
Hybrid similarity scorer that uses different models for different content types:
- Class 0 (Formulas): Mathematical image-to-LaTeX + MathBERT embeddings
- Class 1 (Figures): CLIP visual similarity  
- Class 2 (Tables): CLIP visual similarity
"""

import os
from typing import List, Dict, Tuple

# Import the specialized scorers
from math_similarity_scorer import MathSimilarityScorer

# Import CLIP scorer (restore from backup)
import torch
import torchvision.transforms as transforms
from PIL import Image
import clip
import cv2


class HybridSimilarityScorer:
    """Hybrid scorer using specialized models for different content types."""
    
    def __init__(self, device: str = None):
        """
        Initialize hybrid similarity scorer.
        
        Args:
            device: Device to run models on ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize specialized scorers
        self.math_scorer = MathSimilarityScorer(device)
        
        # CLIP for figures and tables
        self.clip_model = None
        self.image_transform = None
    
    def load_clip_model(self):
        """Load CLIP model for figures and tables."""
        if self.clip_model is None:
            try:
                self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
                self.image_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
                print(f"Loaded CLIP model on device: {self.device}")
            except Exception as e:
                print(f"Error loading CLIP model: {e}")
                self.clip_model = "fallback"
    
    def calculate_clip_similarity(self, image1_path: str, image2_path: str) -> float:
        """Calculate CLIP similarity between two images."""
        self.load_clip_model()
        
        if self.clip_model == "fallback":
            return self._visual_similarity_fallback(image1_path, image2_path)
        
        try:
            # Load and preprocess images
            image_1 = self.image_transform(Image.open(image1_path)).unsqueeze(0).to(self.device)
            image_2 = self.image_transform(Image.open(image2_path)).unsqueeze(0).to(self.device)
            
            # Encode images
            with torch.no_grad():
                image_1_encoding = self.clip_model.encode_image(image_1)
                image_2_encoding = self.clip_model.encode_image(image_2)
            
            # Calculate cosine similarity
            similarity_score = torch.nn.functional.cosine_similarity(
                image_1_encoding, image_2_encoding
            ).item()
            
            return similarity_score
            
        except Exception as e:
            print(f"Error in CLIP similarity: {e}")
            return self._visual_similarity_fallback(image1_path, image2_path)
    
    def _visual_similarity_fallback(self, image1_path: str, image2_path: str) -> float:
        """Fallback visual similarity using traditional CV methods."""
        try:
            # Load images
            img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                return 0.0
            
            # Resize to same size
            img1 = cv2.resize(img1, (224, 224))
            img2 = cv2.resize(img2, (224, 224))
            
            # Calculate histogram similarity
            hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
            
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return max(0, similarity)  # Ensure non-negative
            
        except Exception as e:
            print(f"Error in fallback similarity: {e}")
            return 0.0
    
    def calculate_similarity_by_class(self, image1_path: str, image2_path: str, class_type: str) -> float:
        """
        Calculate similarity based on content class type.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image  
            class_type: Content class ('0' for formulas, '1' for figures, '2' for tables)
            
        Returns:
            Similarity score between 0 and 1
        """
        if class_type == "0":  # Formulas
            print(f"Using mathematical similarity for formulas")
            return self.math_scorer.calculate_math_similarity(image1_path, image2_path)
        elif class_type in ["1", "2"]:  # Figures and Tables  
            print(f"Using CLIP similarity for class {class_type}")
            return self.calculate_clip_similarity(image1_path, image2_path)
        else:
            print(f"Unknown class {class_type}, using CLIP as fallback")
            return self.calculate_clip_similarity(image1_path, image2_path)
    
    def find_best_match(self, reference_image: str, candidate_images: List[str], class_type: str) -> Tuple[str, float, int]:
        """
        Find the best matching image using appropriate similarity method.
        
        Args:
            reference_image: Path to reference image
            candidate_images: List of candidate image paths
            class_type: Content class type
            
        Returns:
            Tuple of (best_match_path, similarity_score, index)
        """
        if not candidate_images:
            return None, 0.0, -1
        
        similarity_scores = []
        
        for candidate_image in candidate_images:
            score = self.calculate_similarity_by_class(reference_image, candidate_image, class_type)
            similarity_scores.append(score)
        
        # Find maximum similarity
        max_index = max(enumerate(similarity_scores), key=lambda x: x[1])[0]
        max_score = similarity_scores[max_index]
        best_match_path = candidate_images[max_index]
        
        return best_match_path, max_score, max_index
    
    def compare_image_sets(self, reference_dir: str, student_dir: str, class_type: str,
                          manual_scores: List[float] = None,
                          display_matches: bool = False,
                          similarity_threshold: float = 0.85) -> Dict:
        """
        Compare all images in reference directory with student directory using appropriate method.
        
        Args:
            reference_dir: Directory containing reference images
            student_dir: Directory containing student images
            class_type: Content class type
            manual_scores: List of manual scoring weights for each question
            display_matches: Whether to display matched image pairs
            similarity_threshold: Minimum similarity score to award full marks
            
        Returns:
            Dictionary containing comparison results
        """
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
            manual_scores.extend([3.0] * (len(reference_images) - len(manual_scores)))
        
        results = {
            'individual_scores': [],
            'similarity_scores': [],
            'total_score': 0.0,
            'matches': []
        }
        
        student_image_paths = [os.path.join(student_dir, img) for img in student_images]
        
        # Class-specific scoring message
        class_names = {"0": "Formulas (Math)", "1": "Figures (CLIP)", "2": "Tables (CLIP)"}
        class_name = class_names.get(class_type, f"Class {class_type}")
        print(f"\nScoring {class_name} using specialized method...")
        
        for i, ref_image in enumerate(reference_images):
            ref_path = os.path.join(reference_dir, ref_image)
            
            # Find best match using appropriate method
            best_match_path, max_similarity, match_index = self.find_best_match(
                ref_path, student_image_paths, class_type
            )
            
            if best_match_path:
                # Apply threshold-based scoring
                if max_similarity >= similarity_threshold:
                    weighted_score = manual_scores[i]
                    score_status = "PASS"
                else:
                    weighted_score = 0.0
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
        
        print(f"\n{class_name} Individual Scores: {results['individual_scores']}")
        print(f"{class_name} Total Score: {results['total_score']:.2f}")
        
        return results
    
    def _display_match(self, ref_path: str, student_path: str):
        """Display matched image pairs."""
        try:
            print(f"Reference: {ref_path}")
            print(f"Student: {student_path}")
            print("---")
        except Exception as e:
            print(f"Error displaying images: {e}")
    
    def batch_compare_classes(self, reference_base_dir: str, student_base_dir: str,
                             class_names: List[str], manual_scores: List[float] = None,
                             similarity_threshold: float = 0.85) -> Dict:
        """
        Compare multiple classes using appropriate specialized methods.
        
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
                    ref_class_dir, student_class_dir, class_name, manual_scores, 
                    similarity_threshold=similarity_threshold
                )
                all_results[class_name] = results
            else:
                print(f"Skipping class {class_name} - directories not found")
                all_results[class_name] = {'total_score': 0.0, 'individual_scores': []}
        
        return all_results