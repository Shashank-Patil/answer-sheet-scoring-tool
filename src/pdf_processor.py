"""
PDF processing utilities for converting PDF files to images.
"""

import os
from typing import List
from pdf2image import convert_from_path
from PIL import Image


class PDFProcessor:
    """Handles PDF to image conversion for answer sheets."""
    
    def __init__(self):
        """Initialize PDF processor."""
        pass
    
    def convert_pdf_to_images(self, pdf_path: str, output_dir: str) -> List[str]:
        """
        Convert PDF file to individual page images.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save converted images
            
        Returns:
            List of paths to saved image files
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert PDF to images
        images = convert_from_path(pdf_path)
        image_paths = []
        
        for i, image in enumerate(images):
            image_path = os.path.join(output_dir, f'page_{i+1}.jpg')
            image.save(image_path, 'JPEG')
            image_paths.append(image_path)
            print(f'Saved {image_path}')
            
        return image_paths
    
    def batch_convert_pdfs(self, pdf_configs: List[dict]) -> dict:
        """
        Convert multiple PDF files to images.
        
        Args:
            pdf_configs: List of dicts with 'pdf_path' and 'output_dir' keys
            
        Returns:
            Dictionary mapping PDF paths to their converted image paths
        """
        results = {}
        
        for config in pdf_configs:
            pdf_path = config['pdf_path']
            output_dir = config['output_dir']
            
            try:
                image_paths = self.convert_pdf_to_images(pdf_path, output_dir)
                results[pdf_path] = image_paths
                print(f"Successfully converted {pdf_path} to {len(image_paths)} images")
            except Exception as e:
                print(f"Error converting {pdf_path}: {e}")
                results[pdf_path] = []
                
        return results