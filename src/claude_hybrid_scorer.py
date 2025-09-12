"""
Claude-powered hybrid similarity scorer.
Uses Claude Sonnet for intelligent comparison of all content types.
"""

import os
from typing import List, Dict, Tuple
from claude_similarity_scorer import ClaudeSimilarityScorer


class ClaudeHybridScorer:
    """Hybrid scorer using Claude Sonnet for all content types with specialized prompts."""
    
    def __init__(self, aws_region: str = "us-east-1"):
        """
        Initialize Claude hybrid scorer.
        
        Args:
            aws_region: AWS region for Bedrock service
        """
        self.claude_scorer = ClaudeSimilarityScorer(aws_region)
        
        # Content type mappings
        self.content_types = {
            "0": "formula",
            "1": "figure", 
            "2": "table"
        }
    
    def compare_image_sets(self, reference_dir: str, student_dir: str, class_type: str,
                          manual_scores: List[float] = None,
                          display_matches: bool = False,
                          similarity_threshold: float = 0.85) -> Dict:
        """
        Compare images using Claude with content-type specific prompts.
        
        Args:
            reference_dir: Directory containing reference images
            student_dir: Directory containing student images
            class_type: Content class type ('0', '1', '2')
            manual_scores: List of manual scoring weights
            display_matches: Whether to display matches (not used for Claude)
            similarity_threshold: Minimum similarity score to award full marks
            
        Returns:
            Dictionary containing comparison results
        """
        # Map class type to content type
        content_type = self.content_types.get(class_type, "unknown")
        
        # Use Claude to compare images
        return self.claude_scorer.compare_image_sets(
            reference_dir=reference_dir,
            student_dir=student_dir,
            content_type=content_type,
            manual_scores=manual_scores,
            similarity_threshold=similarity_threshold
        )
    
    def batch_compare_classes(self, reference_base_dir: str, student_base_dir: str,
                             class_names: List[str], manual_scores: List[float] = None,
                             similarity_threshold: float = 0.85) -> Dict:
        """
        Compare multiple classes using Claude with appropriate content-type prompts.
        
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
        
        class_display_names = {
            "0": "📐 Formulas",
            "1": "📊 Figures", 
            "2": "📋 Tables"
        }
        
        for class_name in class_names:
            ref_class_dir = os.path.join(reference_base_dir, class_name)
            student_class_dir = os.path.join(student_base_dir, class_name)
            
            if os.path.exists(ref_class_dir) and os.path.exists(student_class_dir):
                display_name = class_display_names.get(class_name, f"Class {class_name}")
                print(f"\n{'='*50}")
                print(f"🤖 CLAUDE ANALYSIS: {display_name}")
                print(f"{'='*50}")
                
                results = self.compare_image_sets(
                    ref_class_dir, student_class_dir, class_name, manual_scores, 
                    similarity_threshold=similarity_threshold
                )
                all_results[class_name] = results
            else:
                print(f"⚠️  Skipping class {class_name} - directories not found")
                all_results[class_name] = {'total_score': 0.0, 'individual_scores': []}
        
        return all_results