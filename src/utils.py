"""
Utility functions for the answer sheet scoring tool.
"""

import os
import shutil
from typing import List, Dict, Any
import json


def create_directory(path: str, exist_ok: bool = True) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        path: Directory path to create
        exist_ok: If True, don't raise error if directory exists
    """
    os.makedirs(path, exist_ok=exist_ok)


def get_image_files(directory: str) -> List[str]:
    """
    Get all image files from a directory.
    
    Args:
        directory: Directory to search for images
        
    Returns:
        List of image file paths
    """
    if not os.path.exists(directory):
        return []
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = []
    
    for file in os.listdir(directory):
        if file.lower().endswith(valid_extensions):
            image_files.append(os.path.join(directory, file))
    
    return sorted(image_files)


def copy_files_to_directory(source_files: List[str], target_dir: str) -> List[str]:
    """
    Copy multiple files to a target directory.
    
    Args:
        source_files: List of source file paths
        target_dir: Target directory path
        
    Returns:
        List of copied file paths
    """
    create_directory(target_dir)
    copied_files = []
    
    for source_file in source_files:
        if os.path.exists(source_file):
            filename = os.path.basename(source_file)
            target_path = os.path.join(target_dir, filename)
            shutil.copy2(source_file, target_path)
            copied_files.append(target_path)
    
    return copied_files


def save_results_to_json(results: Dict[str, Any], output_path: str) -> None:
    """
    Save results dictionary to JSON file.
    
    Args:
        results: Results dictionary to save
        output_path: Path to save the JSON file
    """
    create_directory(os.path.dirname(output_path))
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {output_path}")


def load_results_from_json(input_path: str) -> Dict[str, Any]:
    """
    Load results dictionary from JSON file.
    
    Args:
        input_path: Path to the JSON file
        
    Returns:
        Results dictionary
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Results file not found: {input_path}")
    
    with open(input_path, 'r') as f:
        results = json.load(f)
    
    return results


def print_scoring_summary(results: Dict[str, Any]) -> None:
    """
    Print a formatted summary of scoring results.
    
    Args:
        results: Results dictionary from similarity scoring
    """
    print("\n" + "="*50)
    print("SCORING SUMMARY")
    print("="*50)
    
    if isinstance(results, dict) and 'total_score' in results:
        # Single class results
        _print_single_class_summary(results)
    else:
        # Multi-class results
        _print_multi_class_summary(results)


def _print_single_class_summary(results: Dict[str, Any]) -> None:
    """Print summary for single class results."""
    individual_scores = results.get('individual_scores', [])
    similarity_scores = results.get('similarity_scores', [])
    total_score = results.get('total_score', 0.0)
    
    print(f"Individual Scores: {[f'{score:.2f}' for score in individual_scores]}")
    print(f"Similarity Scores: {[f'{score:.4f}' for score in similarity_scores]}")
    print(f"Total Score: {total_score:.2f}")
    
    if 'matches' in results:
        print(f"Number of Matches: {len(results['matches'])}")


def _print_multi_class_summary(results: Dict[str, Dict[str, Any]]) -> None:
    """Print summary for multi-class results."""
    total_all_classes = 0.0
    
    for class_name, class_results in results.items():
        class_total = class_results.get('total_score', 0.0)
        total_all_classes += class_total
        
        print(f"\nClass: {class_name}")
        print(f"  Total Score: {class_total:.2f}")
        
        if 'individual_scores' in class_results:
            individual = class_results['individual_scores']
            print(f"  Individual Scores: {[f'{score:.2f}' for score in individual]}")
    
    print(f"\nGRAND TOTAL: {total_all_classes:.2f}")


def validate_paths(paths: List[str]) -> Dict[str, bool]:
    """
    Validate if all provided paths exist.
    
    Args:
        paths: List of file/directory paths to validate
        
    Returns:
        Dictionary mapping paths to their existence status
    """
    validation_results = {}
    
    for path in paths:
        validation_results[path] = os.path.exists(path)
        if not validation_results[path]:
            print(f"Warning: Path does not exist: {path}")
    
    return validation_results


def clean_directory(directory: str, keep_subdirs: bool = True) -> None:
    """
    Clean all files from a directory, optionally keeping subdirectories.
    
    Args:
        directory: Directory to clean
        keep_subdirs: If True, keep subdirectories but clean their contents
    """
    if not os.path.exists(directory):
        return
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            if keep_subdirs:
                clean_directory(item_path, keep_subdirs=True)
            else:
                shutil.rmtree(item_path)


def get_project_root() -> str:
    """
    Get the project root directory.
    
    Returns:
        Path to project root directory
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)  # Go up one level from src/