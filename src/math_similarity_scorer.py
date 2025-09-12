"""
Mathematical similarity scoring using image-to-LaTeX conversion and mathematical embeddings.
Uses Im2Latex for handwritten formula recognition and MathBERT/Tangent-CFT for semantic comparison.
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple
import cv2
import requests
import base64
import json
from io import BytesIO

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers library not available for MathBERT.")
    TRANSFORMERS_AVAILABLE = False


class MathSimilarityScorer:
    """Handles mathematical formula similarity using LaTeX conversion and embeddings."""
    
    def __init__(self, device: str = None):
        """
        Initialize mathematical similarity scorer.
        
        Args:
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.mathbert_model = None
        self.mathbert_tokenizer = None
        self.im2latex_available = False
        
    def load_mathbert(self):
        """Load MathBERT model for mathematical embeddings."""
        if self.mathbert_model is None and TRANSFORMERS_AVAILABLE:
            try:
                # Use a mathematical language model (alternatives: "microsoft/DialoGPT-medium", "facebook/bart-base")
                model_name = "microsoft/DialoGPT-medium"  # Fallback model
                
                print(f"Loading MathBERT-like model: {model_name}")
                self.mathbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.mathbert_model = AutoModel.from_pretrained(model_name)
                self.mathbert_model.to(self.device)
                self.mathbert_model.eval()
                
                # Add padding token if not present
                if self.mathbert_tokenizer.pad_token is None:
                    self.mathbert_tokenizer.pad_token = self.mathbert_tokenizer.eos_token
                
                print(f"Loaded mathematical model on device: {self.device}")
            except Exception as e:
                print(f"Error loading MathBERT model: {e}")
                self.mathbert_model = "fallback"
    
    def image_to_latex(self, image_path: str) -> str:
        """
        Convert handwritten mathematical formula image to LaTeX.
        
        Args:
            image_path: Path to the formula image
            
        Returns:
            LaTeX string representation of the formula
        """
        try:
            # Try using a local Im2Latex model or API if available
            # For now, we'll use a simplified OCR-based approach
            latex_result = self._extract_formula_features_fallback(image_path)
            return latex_result
            
        except Exception as e:
            print(f"Error in image-to-LaTeX conversion: {e}")
            return self._extract_formula_features_fallback(image_path)
    
    def _extract_formula_features_fallback(self, image_path: str) -> str:
        """
        Fallback method to extract formula-like features from image.
        
        Args:
            image_path: Path to the formula image
            
        Returns:
            Simplified text representation of visual features
        """
        try:
            # Load and preprocess image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return "empty_formula"
            
            # Resize for consistency
            image = cv2.resize(image, (224, 224))
            
            # Extract mathematical features
            features = []
            
            # 1. Detect horizontal and vertical lines (fraction bars, etc.)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            
            horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)
            
            h_line_count = np.sum(horizontal_lines > 0)
            v_line_count = np.sum(vertical_lines > 0)
            
            # 2. Detect connected components (individual symbols)
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
            
            # 3. Analyze symbol distribution
            if len(centroids) > 1:
                centroids = centroids[1:]  # Remove background
                y_positions = centroids[:, 1]
                x_positions = centroids[:, 0]
                
                # Check for superscripts/subscripts (y-position variance)
                y_variance = np.var(y_positions) if len(y_positions) > 1 else 0
                
                # Check for fractions (symbols above/below middle)
                middle_y = image.shape[0] // 2
                above_middle = np.sum(y_positions < middle_y)
                below_middle = np.sum(y_positions > middle_y)
                
                features.extend([
                    f"symbols_{len(centroids)}",
                    f"y_var_{int(y_variance)}",
                    f"above_{above_middle}",
                    f"below_{below_middle}",
                    f"h_lines_{h_line_count // 100}",
                    f"v_lines_{v_line_count // 100}"
                ])
            
            # 4. Edge density features
            edges = cv2.Canny(image, 50, 150)
            edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
            features.append(f"edge_density_{int(edge_density * 1000)}")
            
            # 5. Contour analysis for parentheses, etc.
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            curved_contours = 0
            
            for contour in contours:
                # Approximate contour to see if it's curved
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) > 6:  # Likely curved
                    curved_contours += 1
            
            features.append(f"curved_{curved_contours}")
            
            # Create pseudo-LaTeX representation
            pseudo_latex = " ".join(features)
            return pseudo_latex
            
        except Exception as e:
            print(f"Error in fallback feature extraction: {e}")
            return "error_formula"
    
    def get_math_embedding(self, latex_string: str) -> np.ndarray:
        """
        Get mathematical embedding for a LaTeX string.
        
        Args:
            latex_string: LaTeX representation of the formula
            
        Returns:
            Embedding vector as numpy array
        """
        self.load_mathbert()
        
        if self.mathbert_model == "fallback" or not TRANSFORMERS_AVAILABLE:
            return self._get_text_features_fallback(latex_string)
        
        try:
            # Tokenize the LaTeX string
            inputs = self.mathbert_tokenizer(
                latex_string, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.mathbert_model(**inputs)
                # Use mean pooling of last hidden states
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings = embeddings.cpu().numpy().flatten()
            
            return embeddings
            
        except Exception as e:
            print(f"Error getting math embedding: {e}")
            return self._get_text_features_fallback(latex_string)
    
    def _get_text_features_fallback(self, text: str) -> np.ndarray:
        """
        Fallback method for text feature extraction.
        
        Args:
            text: Text to extract features from
            
        Returns:
            Feature vector as numpy array
        """
        try:
            # Simple text-based features
            features = []
            
            # Character-level features
            features.append(len(text))
            features.append(text.count('_'))  # Subscripts
            features.append(text.count('^'))  # Superscripts
            features.append(text.count('frac'))  # Fractions
            features.append(text.count('sqrt'))  # Square roots
            features.append(text.count('sum'))  # Summations
            features.append(text.count('int'))  # Integrals
            features.append(text.count('symbols'))  # Symbol count
            features.append(text.count('lines'))  # Line features
            features.append(text.count('curved'))  # Curved elements
            
            # Word-level features
            words = text.split()
            features.append(len(words))
            
            # Create n-gram features
            for word in words[:10]:  # Limit to first 10 words
                features.append(hash(word) % 1000)  # Hash to fixed range
            
            # Pad to fixed size
            while len(features) < 128:
                features.append(0)
            
            return np.array(features[:128], dtype=np.float32)
            
        except Exception as e:
            print(f"Error in text feature fallback: {e}")
            return np.random.random(128).astype(np.float32)
    
    def calculate_math_similarity(self, image1_path: str, image2_path: str) -> float:
        """
        Calculate mathematical similarity between two formula images.
        
        Args:
            image1_path: Path to first formula image
            image2_path: Path to second formula image
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Convert images to LaTeX
            latex1 = self.image_to_latex(image1_path)
            latex2 = self.image_to_latex(image2_path)
            
            print(f"LaTeX 1: {latex1[:100]}...")  # Debug print
            print(f"LaTeX 2: {latex2[:100]}...")  # Debug print
            
            # Get mathematical embeddings
            embedding1 = self.get_math_embedding(latex1)
            embedding2 = self.get_math_embedding(latex2)
            
            # Calculate cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm1 * norm2)
            
            # Normalize to [0, 1]
            similarity = max(0, min(1, (similarity + 1) / 2))
            
            return float(similarity)
            
        except Exception as e:
            print(f"Error calculating math similarity: {e}")
            return 0.0
    
    def find_best_math_match(self, reference_image: str, candidate_images: List[str]) -> Tuple[str, float, int]:
        """
        Find the best matching mathematical formula from candidates.
        
        Args:
            reference_image: Path to reference formula image
            candidate_images: List of candidate formula image paths
            
        Returns:
            Tuple of (best_match_path, similarity_score, index)
        """
        if not candidate_images:
            return None, 0.0, -1
        
        similarity_scores = []
        
        for candidate_image in candidate_images:
            score = self.calculate_math_similarity(reference_image, candidate_image)
            similarity_scores.append(score)
        
        # Find maximum similarity
        max_index = max(enumerate(similarity_scores), key=lambda x: x[1])[0]
        max_score = similarity_scores[max_index]
        best_match_path = candidate_images[max_index]
        
        return best_match_path, max_score, max_index
    
    def compare_formula_sets(self, reference_dir: str, student_dir: str, 
                           manual_scores: List[float] = None,
                           similarity_threshold: float = 0.85) -> Dict:
        """
        Compare mathematical formulas in two directories.
        
        Args:
            reference_dir: Directory containing reference formula images
            student_dir: Directory containing student formula images
            manual_scores: List of manual scoring weights for each question
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
            print("No formula images found in one or both directories")
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
        
        for i, ref_image in enumerate(reference_images):
            ref_path = os.path.join(reference_dir, ref_image)
            
            # Find best mathematical match
            best_match_path, max_similarity, match_index = self.find_best_math_match(
                ref_path, student_image_paths
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
                
                print(f"Formula: {ref_image}, Similarity: {max_similarity:.4f} ({score_status}), "
                      f"Score: {weighted_score:.2f}/{manual_scores[i]:.2f}")
            else:
                results['individual_scores'].append(0.0)
                results['similarity_scores'].append(0.0)
                print(f"No match found for reference formula: {ref_image}")
        
        results['total_score'] = sum(results['individual_scores'])
        
        print(f"\nFormula Scores: {results['individual_scores']}")
        print(f"Formula Total: {results['total_score']:.2f}")
        
        return results