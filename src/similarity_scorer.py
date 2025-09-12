"""
Similarity scoring using CLIP model for answer comparison.
"""

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import clip
from typing import List, Dict, Tuple
import cv2


class SimilarityScorer:
    """Handles CLIP-based similarity scoring between answer images."""
    
    def __init__(self, device: str = None):
        """
        Initialize similarity scorer with CLIP model.
        
        Args:
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = None
        self.image_transform = None
    
    def load_model(self):
        """Load the CLIP model."""
        if self.model is None:
            self.model, _ = clip.load("ViT-B/32", device=self.device)
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            print(f"Loaded CLIP model on device: {self.device}")
    
    def calculate_similarity_score(self, image1_path: str, image2_path: str) -> float:
        """
        Calculate similarity score between two images using CLIP.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        self.load_model()
        
        # Load and preprocess images
        image_1 = self.image_transform(Image.open(image1_path)).unsqueeze(0).to(self.device)
        image_2 = self.image_transform(Image.open(image2_path)).unsqueeze(0).to(self.device)
        
        # Encode images
        with torch.no_grad():
            image_1_encoding = self.model.encode_image(image_1)
            image_2_encoding = self.model.encode_image(image_2)
        
        # Calculate cosine similarity
        similarity_score = torch.nn.functional.cosine_similarity(
            image_1_encoding, image_2_encoding
        ).item()
        
        return similarity_score
    
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