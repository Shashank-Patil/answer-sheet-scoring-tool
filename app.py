"""
Streamlit web interface for the Answer Sheet Scoring Tool.
Provides a user-friendly UI for uploading PDFs and running the scoring pipeline.
"""

import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path
import json
from PIL import Image
import pandas as pd

# Import our custom modules
import sys
sys.path.append('src')

try:
    from src.main import AnswerSheetScorer
    from src.utils import get_image_files, print_scoring_summary
except ImportError:
    st.error("Could not import required modules. Make sure you're running from the project root directory.")
    st.stop()


def main():
    st.set_page_config(
        page_title="Answer Sheet Scoring Tool",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📝 Answer Sheet Scoring Tool")
    st.markdown("Automated scoring system using computer vision and deep learning")
    
    # Sidebar for file uploads and configuration
    with st.sidebar:
        st.header("📁 Upload Files")
        
        # File uploads
        reference_pdf = st.file_uploader(
            "Reference Answer Key PDF",
            type=['pdf'],
            help="Upload the reference/template answer key PDF"
        )
        
        student_pdf = st.file_uploader(
            "Student Answer Sheet PDF", 
            type=['pdf'],
            help="Upload the student's answer sheet PDF"
        )
        
        model_file = st.file_uploader(
            "YOLO Model File (best.pt)",
            type=['pt'],
            help="Upload the trained YOLO model file"
        )
        
        st.header("⚙️ Configuration")
        
        # Manual scores configuration
        st.subheader("Manual Scoring Weights")
        num_questions = st.number_input("Number of Questions", min_value=1, max_value=20, value=11)
        
        use_default_scores = st.checkbox("Use default scores (3.0 for all questions)", value=True)
        
        if use_default_scores:
            manual_scores = [3.0] * num_questions
            st.info(f"Using default score of 3.0 for all {num_questions} questions")
        else:
            manual_scores = []
            st.write("Enter score for each question:")
            cols = st.columns(3)
            for i in range(num_questions):
                with cols[i % 3]:
                    score = st.number_input(f"Q{i+1}", min_value=0.0, max_value=10.0, value=3.0, key=f"score_{i}")
                    manual_scores.append(score)
        
        # Confidence threshold
        confidence_threshold = st.slider("Detection Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
        
        # Similarity threshold
        similarity_threshold = st.slider("Similarity Threshold (for full marks)", 0.1, 1.0, 0.85, 0.05)
        st.info(f"Answers with similarity ≥ {similarity_threshold:.2f} get full marks, below get 0")
        
        # Run button
        run_scoring = st.button("🚀 Run Scoring Pipeline", type="primary", use_container_width=True)
    
    # Main content area
    if not run_scoring:
        show_welcome_screen()
    else:
        if not all([reference_pdf, student_pdf, model_file]):
            st.error("❌ Please upload all required files (Reference PDF, Student PDF, and Model file)")
            return
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save uploaded files
            ref_pdf_path = temp_path / "reference.pdf"
            student_pdf_path = temp_path / "student.pdf"
            model_path = temp_path / "best.pt"
            
            with open(ref_pdf_path, "wb") as f:
                f.write(reference_pdf.read())
            with open(student_pdf_path, "wb") as f:
                f.write(student_pdf.read())
            with open(model_path, "wb") as f:
                f.write(model_file.read())
            
            run_scoring_pipeline(str(ref_pdf_path), str(student_pdf_path), str(model_path), 
                               manual_scores, temp_dir, confidence_threshold, similarity_threshold)


def show_welcome_screen():
    """Display welcome screen with instructions and examples."""
    st.header("🎯 How to Use")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Required Files")
        st.markdown("""
        1. **Reference Answer Key PDF**: The correct/template answers
        2. **Student Answer Sheet PDF**: The student's submitted answers  
        3. **YOLO Model File**: Pre-trained model for object detection (`best.pt`)
        """)
        
        st.subheader("⚡ Quick Start")
        st.markdown("""
        1. Upload all three required files in the sidebar
        2. Configure scoring weights (or use defaults)
        3. Click "Run Scoring Pipeline"
        4. View results with visual comparisons
        """)
    
    with col2:
        st.subheader("🔧 What It Does")
        st.markdown("""
        - **PDF to Images**: Converts PDFs to individual page images
        - **Object Detection**: Uses YOLO to detect answer regions
          - 📐 **Class 0**: Formulas/Equations
          - 📊 **Class 1**: Figures/Diagrams
          - 📋 **Class 2**: Tables/Structured Data
        - **Similarity Scoring**: Uses Claude Sonnet 4 via AWS Bedrock
          - 🤖 **Intelligent Analysis**: Understands mathematical formulas, diagrams, and tables
          - 📐 **Formulas**: Semantic mathematical comparison (not just visual)
          - 📊 **Figures**: Structural and content analysis
          - 📋 **Tables**: Data organization and value comparison
        - **Automated Grading**: Threshold-based scoring (≥0.85 = full marks)
        """)
        
        st.subheader("📊 Output")
        st.markdown("""
        - Individual question scores
        - Similarity comparisons  
        - Visual previews of detected regions
        - Final total score
        """)
    
    # Example section
    st.header("📸 Example Output Preview")
    st.markdown("Here's what you can expect to see after processing:")
    
    # Create example data
    example_data = {
        "Question": [f"Q{i}" for i in range(1, 6)],
        "Similarity Score": [0.85, 0.92, 0.78, 0.88, 0.95],
        "Manual Weight": [3.0, 3.0, 3.0, 3.0, 3.0],
        "Final Score": [2.55, 2.76, 2.34, 2.64, 2.85]
    }
    
    df = pd.DataFrame(example_data)
    st.dataframe(df, use_container_width=True)
    
    st.success("📈 Total Score: 12.14 / 15.00")


def run_scoring_pipeline(ref_pdf_path, student_pdf_path, model_path, manual_scores, temp_dir, confidence, similarity_threshold):
    """Run the complete scoring pipeline with progress tracking."""
    
    st.header("🔄 Processing Pipeline")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize scorer
        status_text.text("Initializing scorer...")
        output_dir = os.path.join(temp_dir, "outputs")
        scorer = AnswerSheetScorer(model_path, output_dir)
        progress_bar.progress(10)
        
        # Step 1: PDF Processing
        status_text.text("Converting PDFs to images...")
        st.subheader("📄 Step 1: PDF Processing")
        
        with st.expander("View PDF Conversion Details", expanded=False):
            image_dirs = scorer.process_pdfs(ref_pdf_path, student_pdf_path)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Reference PDF Pages:**")
                if image_dirs['reference']:
                    st.success(f"✅ Converted {len(image_dirs['reference'])} pages")
                    # Show first page preview
                    if image_dirs['reference']:
                        img = Image.open(image_dirs['reference'][0])
                        st.image(img, caption="Reference Page 1 Preview", width=300)
                
            with col2:
                st.write("**Student PDF Pages:**")
                if image_dirs['student']:
                    st.success(f"✅ Converted {len(image_dirs['student'])} pages")
                    # Show first page preview
                    if image_dirs['student']:
                        img = Image.open(image_dirs['student'][0])
                        st.image(img, caption="Student Page 1 Preview", width=300)
        
        progress_bar.progress(30)
        
        # Step 2: Object Detection
        status_text.text("Running object detection...")
        st.subheader("🎯 Step 2: Object Detection")
        
        with st.expander("View Detection Results", expanded=True):
            detection_results = scorer.detect_objects(image_dirs)
            
            # Class mapping
            class_mapping = {
                "0": "📐 Formulas/Equations",
                "1": "📊 Figures/Diagrams", 
                "2": "📋 Tables/Structured Data"
            }
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Reference Detections:**")
                ref_crops = detection_results['reference']['crops']
                for class_name, crops in ref_crops.items():
                    class_display = class_mapping.get(class_name, f"Class {class_name}")
                    st.write(f"- {class_display}: {len(crops)} detections")
                    if crops:  # Show first detection preview
                        img = Image.open(crops[0])
                        st.image(img, caption=f"Reference {class_display} (Preview)", width=250)
            
            with col2:
                st.write("**Student Detections:**")
                student_crops = detection_results['student']['crops']
                for class_name, crops in student_crops.items():
                    class_display = class_mapping.get(class_name, f"Class {class_name}")
                    st.write(f"- {class_display}: {len(crops)} detections")
                    if crops:  # Show first detection preview
                        img = Image.open(crops[0])
                        st.image(img, caption=f"Student {class_display} (Preview)", width=250)
        
        progress_bar.progress(60)
        
        # Step 3: Similarity Scoring
        status_text.text("Calculating similarity scores...")
        st.subheader("🧮 Step 3: Similarity Scoring")
        
        with st.expander("View Scoring Process", expanded=True):
            scoring_results = scorer.calculate_scores(detection_results, manual_scores, similarity_threshold)
            progress_bar.progress(90)
            
            # Class mapping
            class_mapping = {
                "0": "📐 Formulas/Equations",
                "1": "📊 Figures/Diagrams", 
                "2": "📋 Tables/Structured Data"
            }
            
            # Display results for each class
            total_all_classes = 0
            all_individual_scores = []
            
            for class_name, class_results in scoring_results.items():
                class_display = class_mapping.get(class_name, f"Class {class_name}")
                st.write(f"**{class_display} Results:**")
                
                if 'matches' in class_results and class_results['matches']:
                    num_matches = len(class_results['matches'])
                    st.info(f"Showing all {num_matches} comparisons")
                    
                    # Create comparison grid - use max 4 columns for better layout
                    max_cols = 4
                    num_rows = (num_matches + max_cols - 1) // max_cols
                    
                    for row in range(num_rows):
                        cols = st.columns(max_cols)
                        start_idx = row * max_cols
                        end_idx = min(start_idx + max_cols, num_matches)
                        
                        for col_idx, match_idx in enumerate(range(start_idx, end_idx)):
                            match = class_results['matches'][match_idx]
                            
                            with cols[col_idx]:
                                # Reference image
                                ref_img = Image.open(match['reference'])
                                st.image(ref_img, caption=f"Reference {match_idx + 1}", width=180)
                                
                                # Student image
                                student_img = Image.open(match['student'])
                                st.image(student_img, caption=f"Student {match_idx + 1}", width=180)
                                
                                # Scores with pass/fail status
                                passed = match.get('passed_threshold', match['similarity'] >= similarity_threshold)
                                status_color = "🟢" if passed else "🔴"
                                status_label = "PASS" if passed else "FAIL"
                                
                                st.metric(
                                    label="Similarity", 
                                    value=f"{match['similarity']:.3f}",
                                    delta=f"{status_color} {status_label}"
                                )
                                st.write(f"**Score: {match['weighted_score']:.1f}**")
                
                class_total = class_results.get('total_score', 0)
                total_all_classes += class_total
                all_individual_scores.extend(class_results.get('individual_scores', []))
                
                st.write(f"{class_display} Total: **{class_total:.2f}**")
                st.divider()
        
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        # Final Results
        st.header("🎉 Final Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Total Score",
                value=f"{total_all_classes:.2f}",
                delta=f"Max Possible: {sum(manual_scores):.2f}"
            )
        
        with col2:
            percentage = (total_all_classes / sum(manual_scores)) * 100 if sum(manual_scores) > 0 else 0
            st.metric(
                label="Percentage",
                value=f"{percentage:.1f}%"
            )
        
        with col3:
            st.metric(
                label="Questions Scored",
                value=len(all_individual_scores)
            )
        
        # Detailed breakdown
        st.subheader("📊 Detailed Breakdown")
        
        # Class mapping for table
        class_mapping = {
            "0": "📐 Formulas",
            "1": "📊 Figures", 
            "2": "📋 Tables"
        }
        
        # Create detailed results DataFrame
        detailed_results = []
        question_num = 1
        
        for class_name, class_results in scoring_results.items():
            individual_scores = class_results.get('individual_scores', [])
            similarity_scores = class_results.get('similarity_scores', [])
            
            class_display = class_mapping.get(class_name, class_name)
            
            for i, (ind_score, sim_score) in enumerate(zip(individual_scores, similarity_scores)):
                status = "🟢 PASS" if sim_score >= similarity_threshold else "🔴 FAIL"
                detailed_results.append({
                    "Question": f"Q{question_num}",
                    "Type": class_display,
                    "Similarity": f"{sim_score:.4f}",
                    "Status": status,
                    "Weight": manual_scores[min(question_num-1, len(manual_scores)-1)],
                    "Score": f"{ind_score:.1f}"
                })
                question_num += 1
        
        if detailed_results:
            df = pd.DataFrame(detailed_results)
            st.dataframe(df, use_container_width=True)
        
        # Download results
        results_json = json.dumps(scoring_results, indent=2, default=str)
        st.download_button(
            label="📥 Download Results (JSON)",
            data=results_json,
            file_name="scoring_results.json",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()