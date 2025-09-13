# Answer Sheet Scoring Tool

An automated scoring system that uses computer vision and deep learning to evaluate handwritten answer sheets by comparing them with reference answer keys.

## Overview

This tool leverages YOLO object detection and multiple similarity scoring strategies to automatically score student answer sheets. It extracts question regions from both reference answer keys and student submissions, then computes similarity scores to provide automated grading.

## Features

- **PDF to Image Conversion**: Converts PDF answer sheets to images for processing
- **Object Detection**: Uses YOLOv8 to detect and extract question regions from answer sheets
- **Dual Similarity Strategies**:
  - **CLIP** (default): Fast local embeddings for similarity comparison
  - **Nova**: AWS Bedrock vision model with intelligent semantic analysis
- **Automated Grading**: Strategy-based equivalence determination for pass/fail scoring
- **Multi-category Detection**: Handles different types of content (formulas, figures, tables)
- **Web Interface**: User-friendly Streamlit app for easy interaction
- **Smart Caching**: Intelligent caching for Nova strategy (CLIP is fast enough without caching)

## System Requirements

- Python 3.7+
- PyTorch
- OpenCV
- PIL (Pillow)
- pdf2image
- ultralytics (YOLOv8)
- For CLIP strategy: torch, clip-by-openai
- For Nova strategy: boto3 (AWS credentials required)
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

# Install CLIP strategy requirements
pip install clip-by-openai

# Install Nova strategy requirements (optional)
pip install boto3

# Install additional dependencies
pip install numpy>=1.23.5 tqdm>=4.65.0 streamlit

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
2. Choose similarity strategy (CLIP or Nova)
3. Configure scoring weights
4. Click "Run Scoring Pipeline"
5. View results with visual previews and comparisons

#### Option B: Using the Command Line

```bash
# Navigate to the src directory
cd src

# Run with CLIP strategy (default)
python main.py \
  --reference-pdf ../data/PDC_Assignment_3_solutions_2023.pdf \
  --student-pdf "../data/RollNo_PDC_Assignment 3.pdf" \
  --model-path ../models/best.pt \
  --output-dir ../outputs \
  --strategy clip \
  --manual-scores 3 3 3 3 3 3 3 3 3 3 3

# Run with Nova strategy
python main.py \
  --reference-pdf ../data/PDC_Assignment_3_solutions_2023.pdf \
  --student-pdf "../data/RollNo_PDC_Assignment 3.pdf" \
  --model-path ../models/best.pt \
  --output-dir ../outputs \
  --strategy nova \
  --manual-scores 3 3 3 3 3 3 3 3 3 3 3
```

#### Option C: Using Python Script

```python
from src.main import AnswerSheetScorer

# Initialize scorer with CLIP strategy (default)
scorer = AnswerSheetScorer("models/best.pt", "outputs", strategy="clip")

# Or initialize with Nova strategy
scorer = AnswerSheetScorer("models/best.pt", "outputs", strategy="nova")

# Run complete pipeline
results = scorer.run_complete_pipeline(
    "data/PDC_Assignment_3_solutions_2023.pdf",
    "data/RollNo_PDC_Assignment 3.pdf",
    manual_scores=[3]*11
)
```

### 3. Output

The system will:
- Display detected question regions
- Show similarity comparisons between reference and student answers
- Provide individual question scores with strategy explanations
- Calculate total score using strategy-based equivalence determination

## Similarity Strategies

### CLIP Strategy (Default)
- Uses OpenAI's CLIP model for visual similarity
- Fast, local processing (no internet required after model download)
- Good for general visual similarity comparison
- Cosine similarity with normalized scoring

### Nova Strategy
- Uses AWS Bedrock's Nova vision model
- Requires AWS credentials and internet connection
- Advanced semantic understanding of mathematical formulas, diagrams, and tables
- Intelligent equivalence determination with detailed explanations
- Better for complex mathematical and scientific content

## Project Structure

```
answer-sheet-scoring-tool/
├── README.md
├── requirements.txt         # Python dependencies
├── app.py                  # Streamlit web interface
├── .gitignore
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── main.py            # Main pipeline orchestrator
│   ├── pdf_processor.py    # PDF to image conversion
│   ├── object_detector.py  # YOLO object detection
│   ├── similarity_scorer.py # Unified similarity scoring (CLIP + Nova)
│   ├── config.py          # Configuration parameters
│   ├── result_cache.py    # Result caching system
│   └── utils.py           # Utility functions
├── notebooks/             # Jupyter notebooks (if any)
├── data/                 # Input data files
│   ├── PDC_Assignment_3_solutions_2023.pdf  # Reference answer key
│   └── RollNo_PDC_Assignment 3.pdf         # Student submission
├── models/               # Trained models
│   └── best.pt          # Pre-trained YOLO model
├── outputs/             # Generated outputs
├── cache/              # Cached results
└── docs/               # Documentation
```

## How It Works

1. **PDF Processing**: Convert PDF documents to individual page images
2. **Region Detection**: Use YOLO to identify and crop question regions by type
3. **Similarity Analysis**:
   - **CLIP**: Extract visual embeddings and compute cosine similarity
   - **Nova**: Use AWS Bedrock for semantic comparison with detailed reasoning
4. **Score Calculation**: Apply strategy-based equivalence determination with manual weights

## Model Details

- **Object Detection**: YOLOv8 trained to detect different answer types
- **CLIP Strategy**: OpenAI CLIP ViT-B/32 model for visual embeddings
- **Nova Strategy**: AWS Bedrock Nova vision model for semantic analysis
- **Classes**:
  - Class 0: Formulas/equations (📐)
  - Class 1: Figures/diagrams (📊)
  - Class 2: Tables/structured data (📋)

## Configuration

### Manual Scoring Weights
```python
manual_scores = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]  # Weights for each question
```

### Strategy Selection
```bash
# Use CLIP (default, fast)
--strategy clip

# Use Nova (advanced, requires AWS)
--strategy nova
```

### AWS Configuration (for Nova strategy)
```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=ap-south-1
```

## Advanced Features

- **Smart Caching**: Automatically caches Nova results to speed up repeated runs (CLIP is fast enough without caching)
- **Content-Type Aware**: Different handling for formulas, figures, and tables
- **Flexible Scoring**: Both absolute (pass/fail) and partial (similarity-based) scoring
- **Visual Feedback**: Web interface shows detected regions and comparisons
- **Export Results**: Download detailed results in JSON format

## Limitations

- Requires pre-trained YOLO model for specific answer sheet formats
- Performance depends on image quality and handwriting clarity
- Nova strategy requires AWS credentials and internet connection
- CLIP strategy may not understand complex mathematical relationships as well as Nova

## Troubleshooting

### CLIP Strategy Issues
- Ensure PyTorch and clip-by-openai are installed
- First run downloads CLIP model (requires internet)

### Nova Strategy Issues
- Verify AWS credentials are configured
- Check AWS Bedrock access permissions
- Ensure stable internet connection

### General Issues
- Check poppler-utils installation for PDF processing
- Verify YOLO model file exists and is accessible
- Ensure sufficient disk space for image processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with both CLIP and Nova strategies
5. Submit a pull request

## License

This project is open source. Please check the license file for details.

## Contact

For questions or support, please open an issue in the repository.