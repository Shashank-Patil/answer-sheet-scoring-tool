# Answer Sheet Scoring Tool

An automated scoring system that uses computer vision and deep learning to evaluate handwritten answer sheets by comparing them with reference answer keys.

## Overview

This tool leverages YOLO object detection and CLIP similarity matching to automatically score student answer sheets. It extracts question regions from both reference answer keys and student submissions, then computes similarity scores to provide automated grading.

## Features

- **PDF to Image Conversion**: Converts PDF answer sheets to images for processing
- **Object Detection**: Uses YOLOv8 to detect and extract question regions from answer sheets
- **Similarity Scoring**: Employs CLIP (Contrastive Language-Image Pre-training) to compute similarity between reference and student answers
- **Automated Grading**: Combines manual scoring weights with similarity scores for final grades
- **Multi-category Detection**: Handles different types of content (formulas, figures, tables)

## System Requirements

- Python 3.7+
- PyTorch
- OpenCV
- PIL (Pillow)
- pdf2image
- ultralytics (YOLOv8)
- CLIP
- poppler-utils (for PDF processing)

## Installation

### Option 1: Using requirements.txt (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd answer-sheet-scoring-tool

# Install Python dependencies
pip install -r requirements.txt

# Install poppler-utils (Ubuntu/Debian)
sudo apt-get install poppler-utils

# Install poppler-utils (macOS)
brew install poppler
```

### Option 2: Manual Installation

```bash
# Install core packages
pip install pdf2image>=1.16.3 ultralytics>=8.0.152
pip install torch>=2.0.1 torchvision>=0.15.2
pip install opencv-python>=4.8.0 Pillow>=9.4.0

# Install CLIP requirements
pip install ftfy>=6.1.1 regex>=2022.10.31
pip install git+https://github.com/openai/CLIP.git

# Install additional dependencies
pip install numpy>=1.23.5 tqdm>=4.65.0

# Install poppler-utils (Ubuntu/Debian)
sudo apt-get install poppler-utils

# Install poppler-utils (macOS)
brew install poppler
```

## Usage

### 1. Prepare Your Data

Place your files in the appropriate directories:
- Reference answer key PDF → `data/`
- Student answer sheet PDF → `data/`
- Pre-trained YOLO model (`best.pt`) → `models/`

### 2. Run the Analysis

#### Option A: Using Streamlit Web UI (Recommended)

```bash
# Run the web interface
streamlit run app.py
```

Then open your browser to `http://localhost:8501` and:
1. Upload your reference PDF, student PDF, and YOLO model
2. Configure scoring weights
3. Click "Run Scoring Pipeline"
4. View results with visual previews and comparisons

#### Option B: Using the Python Module

```bash
# Navigate to the src directory
cd src

# Run the complete pipeline
python main.py \
  --reference-pdf ../data/PDC_Assignment_3_solutions_2023.pdf \
  --student-pdf "../data/RollNo_PDC_Assignment 3.pdf" \
  --model-path ../models/best.pt \
  --output-dir ../outputs \
  --manual-scores 3 3 3 3 3 3 3 3 3 3 3
```

#### Option C: Using Python Script

```python
from src.main import AnswerSheetScorer

# Initialize scorer
scorer = AnswerSheetScorer("models/best.pt", "outputs")

# Run complete pipeline
results = scorer.run_complete_pipeline(
    "data/PDC_Assignment_3_solutions_2023.pdf",
    "data/RollNo_PDC_Assignment 3.pdf",
    manual_scores=[3]*11
)
```

#### Option D: Using the Jupyter Notebook

Execute the Jupyter notebook `notebooks/DDP_Demo.ipynb` which includes:

1. **PDF Conversion**: Convert PDFs to images
2. **Object Detection**: Extract question regions using YOLO
3. **Similarity Analysis**: Compare student answers with reference answers
4. **Scoring**: Generate final scores based on similarity and manual weights

### 3. Output

The system will:
- Display detected question regions
- Show similarity comparisons between reference and student answers
- Provide individual question scores
- Calculate total score

## Project Structure

```
answer-sheet-scoring-tool/
├── README.md
├── requirements.txt         # Python dependencies
├── .gitignore
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── pdf_processor.py    # PDF to image conversion
│   ├── object_detector.py  # YOLO object detection
│   ├── similarity_scorer.py # CLIP similarity scoring
│   └── utils.py           # Utility functions
├── notebooks/             # Jupyter notebooks
│   └── DDP_Demo.ipynb    # Main analysis notebook
├── data/                 # Input data files
│   ├── PDC_Assignment_3_solutions_2023.pdf  # Reference answer key
│   └── RollNo_PDC_Assignment 3.pdf         # Student submission
├── models/               # Trained models
│   └── best.pt          # Pre-trained YOLO model
├── outputs/             # Generated outputs
├── docs/               # Documentation and presentations
│   ├── DDP_Presentation.pdf
│   └── DDP_Presentation.pptx
└── tests/              # Unit tests (future)
```

## How It Works

1. **PDF Processing**: Convert PDF documents to individual page images
2. **Region Detection**: Use YOLO to identify and crop question regions
3. **Feature Extraction**: Apply CLIP to extract visual features from cropped regions
4. **Similarity Computation**: Calculate cosine similarity between reference and student answers
5. **Score Calculation**: Multiply similarity scores by manual weights to get final scores

## Model Details

- **Object Detection**: YOLOv8 trained to detect different answer types (formulas, figures, tables)
- **Similarity Matching**: CLIP ViT-B/32 model for visual similarity comparison
- **Classes**: 
  - Class 0: Formulas/equations
  - Class 1: Figures/diagrams  
  - Class 2: Tables/structured data

## Configuration

Manual scoring weights can be adjusted in the notebook:
```python
manual_scores = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]  # Weights for each question
```

## Limitations

- Requires pre-trained YOLO model for specific answer sheet formats
- Performance depends on image quality and handwriting clarity
- Manual weight configuration needed for different assignment types

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source. Please check the license file for details.

## Contact

For questions or support, please open an issue in the repository.