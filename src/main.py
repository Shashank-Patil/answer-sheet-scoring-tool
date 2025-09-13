"""
Main entry point for the Answer Sheet Scoring Tool.
Orchestrates the complete pipeline from PDF processing to final scoring.
"""

import os
import argparse
from typing import Dict, List

from pdf_processor import PDFProcessor
from object_detector import ObjectDetector
from similarity_scorer import SimilarityScorer
from utils import (
    create_directory, save_results_to_json, print_scoring_summary,
    validate_paths, get_project_root
)


class AnswerSheetScorer:
    """Main class for the answer sheet scoring pipeline."""
    
    def __init__(self, model_path: str, output_base_dir: str = "outputs",
                 similarity_strategy: str = "clip", enable_cache: bool = True):
        """
        Initialize the answer sheet scorer.

        Args:
            model_path: Path to the trained YOLO model
            output_base_dir: Base directory for all outputs
            similarity_strategy: Similarity strategy ('nova' or 'clip', default: 'clip')
            enable_cache: Whether to enable result caching
        """
        self.model_path = model_path
        self.output_base_dir = output_base_dir

        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.object_detector = ObjectDetector(model_path)
        self.similarity_scorer = SimilarityScorer(strategy=similarity_strategy, enable_cache=enable_cache)

        # Create output directory
        create_directory(self.output_base_dir)
    
    def process_pdfs(self, reference_pdf: str, student_pdf: str) -> Dict[str, List[str]]:
        """
        Convert PDFs to images.
        
        Args:
            reference_pdf: Path to reference answer key PDF
            student_pdf: Path to student answer sheet PDF
            
        Returns:
            Dictionary with 'reference' and 'student' image paths
        """
        print("Converting PDFs to images...")
        
        # Define output directories
        ref_images_dir = os.path.join(self.output_base_dir, "reference_images")
        student_images_dir = os.path.join(self.output_base_dir, "student_images")
        
        # Convert PDFs
        pdf_configs = [
            {'pdf_path': reference_pdf, 'output_dir': ref_images_dir},
            {'pdf_path': student_pdf, 'output_dir': student_images_dir}
        ]
        
        conversion_results = self.pdf_processor.batch_convert_pdfs(pdf_configs)
        
        return {
            'reference': conversion_results.get(reference_pdf, []),
            'student': conversion_results.get(student_pdf, [])
        }
    
    def detect_objects(self, image_dirs: Dict[str, List[str]]) -> Dict[str, Dict]:
        """
        Run object detection on converted images.
        
        Args:
            image_dirs: Dictionary with reference and student image directories
            
        Returns:
            Dictionary with detection results for both reference and student
        """
        print("Running object detection...")
        
        detection_results = {}
        
        # Process reference images
        if image_dirs['reference']:
            ref_source_dir = os.path.dirname(image_dirs['reference'][0])
            ref_output_dir = os.path.join(self.output_base_dir, "reference_detection")
            ref_crops = self.object_detector.process_images(
                ref_source_dir, ref_output_dir, conf=0.25
            )
            detection_results['reference'] = {
                'crops': ref_crops,
                'output_dir': ref_output_dir
            }
        
        # Process student images
        if image_dirs['student']:
            student_source_dir = os.path.dirname(image_dirs['student'][0])
            student_output_dir = os.path.join(self.output_base_dir, "student_detection")
            student_crops = self.object_detector.process_images(
                student_source_dir, student_output_dir, conf=0.25
            )
            detection_results['student'] = {
                'crops': student_crops,
                'output_dir': student_output_dir
            }
        
        return detection_results
    
    def calculate_scores(self, detection_results: Dict[str, Dict], 
                        manual_scores: List[float] = None,
                        similarity_threshold: float = 0.85) -> Dict:
        """
        Calculate similarity scores between reference and student answers.
        
        Args:
            detection_results: Results from object detection
            manual_scores: Manual scoring weights for each question
            similarity_threshold: Minimum similarity score to award full marks
            
        Returns:
            Dictionary with scoring results
        """
        print("Calculating similarity scores...")
        
        if manual_scores is None:
            manual_scores = [3.0] * 11  # Default from notebook
        
        ref_crops = detection_results['reference']['crops']
        student_crops = detection_results['student']['crops']
        
        # Get common classes
        common_classes = set(ref_crops.keys()) & set(student_crops.keys())
        
        if not common_classes:
            print("No common classes found between reference and student")
            return {}
        
        print(f"Found common classes: {list(common_classes)}")
        
        # Calculate scores for each class
        all_results = {}
        
        for class_name in common_classes:
            ref_class_dir = os.path.join(
                detection_results['reference']['output_dir'], 
                'extracted_crops', class_name
            )
            student_class_dir = os.path.join(
                detection_results['student']['output_dir'], 
                'extracted_crops', class_name
            )
            
            print(f"\nScoring class: {class_name}")
            class_results = self.similarity_scorer.compare_image_sets(
                ref_class_dir, student_class_dir, class_name, manual_scores
            )
            all_results[class_name] = class_results
        
        return all_results
    
    def run_complete_pipeline(self, reference_pdf: str, student_pdf: str,
                             manual_scores: List[float] = None,
                             similarity_threshold: float = 0.85,
                             save_results: bool = True) -> Dict:
        """
        Run the complete scoring pipeline.
        
        Args:
            reference_pdf: Path to reference answer key PDF
            student_pdf: Path to student answer sheet PDF
            manual_scores: Manual scoring weights
            similarity_threshold: Minimum similarity score to award full marks
            save_results: Whether to save results to JSON
            
        Returns:
            Dictionary with complete results
        """
        print("Starting Answer Sheet Scoring Pipeline")
        print("="*50)
        
        # Validate input paths
        paths_to_validate = [reference_pdf, student_pdf, self.model_path]
        validation_results = validate_paths(paths_to_validate)
        
        if not all(validation_results.values()):
            raise FileNotFoundError("One or more required files not found")
        
        # Step 1: Convert PDFs to images
        image_dirs = self.process_pdfs(reference_pdf, student_pdf)
        
        # Step 2: Run object detection
        detection_results = self.detect_objects(image_dirs)
        
        # Step 3: Calculate similarity scores
        scoring_results = self.calculate_scores(detection_results, manual_scores, similarity_threshold)
        
        # Print summary
        print_scoring_summary(scoring_results)
        
        # Save results if requested
        if save_results:
            results_path = os.path.join(self.output_base_dir, "scoring_results.json")
            save_results_to_json(scoring_results, results_path)
        
        return scoring_results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Automated Answer Sheet Scoring Tool"
    )
    
    parser.add_argument(
        "--reference-pdf", 
        required=True,
        help="Path to reference answer key PDF"
    )
    
    parser.add_argument(
        "--student-pdf",
        required=True, 
        help="Path to student answer sheet PDF"
    )
    
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to trained YOLO model (best.pt)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for results (default: outputs)"
    )
    
    parser.add_argument(
        "--manual-scores",
        nargs="+",
        type=float,
        help="Manual scoring weights (space-separated floats)"
    )
    
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="DEPRECATED - strategy makes the equivalent/not decision (default: 0.85)"
    )

    parser.add_argument(
        "--strategy",
        choices=["nova", "clip"],
        default="clip",
        help="Similarity strategy: 'nova' for AWS Bedrock Nova or 'clip' for CLIP embeddings (default: clip)"
    )
    
    args = parser.parse_args()
    
    # Initialize scorer
    scorer = AnswerSheetScorer(args.model_path, args.output_dir, args.strategy)
    
    # Run pipeline
    try:
        results = scorer.run_complete_pipeline(
            args.reference_pdf,
            args.student_pdf, 
            args.manual_scores,
            args.similarity_threshold
        )
        print("\nPipeline completed successfully!")
        
    except Exception as e:
        print(f"Error running pipeline: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())