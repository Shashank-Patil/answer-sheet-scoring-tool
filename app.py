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
        
        # Similarity strategy selection
        st.subheader("🧠 Similarity Strategy")
        strategy = st.selectbox(
            "Choose similarity strategy:",
            options=["clip", "nova"],
            index=0,  # Default to CLIP
            help="CLIP: Fast local embeddings (default) | Nova: AWS Bedrock vision model"
        )

        if strategy == "clip":
            st.info("📊 Using CLIP embeddings for similarity comparison")
        else:
            st.info("🤖 Using Nova vision model - automatically determines if answers are equivalent!")
        
        # Cache override option (only show for Nova strategy)
        if strategy == "nova":
            st.subheader("🗄️ Cache Settings")
            bypass_cache = st.checkbox("Force re-analysis (ignore cached results)", value=False,
                                      help="Check this to run fresh Nova analysis even if cached results exist. Useful for testing different prompts or when you want the latest analysis.")
        else:
            bypass_cache = False  # No caching for CLIP strategy
        
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
                               manual_scores, temp_dir, confidence_threshold, strategy, bypass_cache)


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
        - **Similarity Scoring**: Choose between two strategies:
          - 📊 **CLIP** (default): Fast local embeddings for similarity comparison
          - 🤖 **Nova**: AWS Bedrock vision model with intelligent analysis
            - Understands mathematical formulas, diagrams, and tables
            - 📐 **Formulas**: Semantic mathematical comparison (not just visual)
            - 📊 **Figures**: Structural and content analysis
            - 📋 **Tables**: Data organization and value comparison
        - **Automated Grading**: Strategy determines equivalence (intelligent pass/fail)
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


def run_scoring_pipeline(ref_pdf_path, student_pdf_path, model_path, manual_scores, temp_dir, confidence, strategy, bypass_cache=False):
    """Run the complete scoring pipeline with progress tracking."""
    
    st.header("🔄 Processing Pipeline")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize scorer
        status_text.text("Initializing scorer...")
        output_dir = os.path.join(temp_dir, "outputs")
        scorer = AnswerSheetScorer(model_path, output_dir, strategy, enable_cache=not bypass_cache)
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
                    # Image carousel for reference pages
                    if len(image_dirs['reference']) > 1:
                        ref_page_idx = st.selectbox(
                            "Select Reference Page", 
                            range(len(image_dirs['reference'])),
                            format_func=lambda x: f"Page {x+1}",
                            key="ref_page_selector"
                        )
                    else:
                        ref_page_idx = 0
                    
                    if image_dirs['reference']:
                        img = Image.open(image_dirs['reference'][ref_page_idx])
                        st.image(img, caption=f"Reference Page {ref_page_idx + 1}", width=300)
                
            with col2:
                st.write("**Student PDF Pages:**")
                if image_dirs['student']:
                    st.success(f"✅ Converted {len(image_dirs['student'])} pages")
                    # Image carousel for student pages
                    if len(image_dirs['student']) > 1:
                        student_page_idx = st.selectbox(
                            "Select Student Page", 
                            range(len(image_dirs['student'])),
                            format_func=lambda x: f"Page {x+1}",
                            key="student_page_selector"
                        )
                    else:
                        student_page_idx = 0
                    
                    if image_dirs['student']:
                        img = Image.open(image_dirs['student'][student_page_idx])
                        st.image(img, caption=f"Student Page {student_page_idx + 1}", width=300)
        
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
                    if crops:
                        if len(crops) > 1:
                            ref_crop_idx = st.selectbox(
                                f"Select {class_display}", 
                                range(len(crops)),
                                format_func=lambda x: f"Detection {x+1}",
                                key=f"ref_{class_name}_selector"
                            )
                        else:
                            ref_crop_idx = 0
                        
                        img = Image.open(crops[ref_crop_idx])
                        st.image(img, caption=f"Reference {class_display} ({ref_crop_idx + 1}/{len(crops)})", width=350)
            
            with col2:
                st.write("**Student Detections:**")
                student_crops = detection_results['student']['crops']
                for class_name, crops in student_crops.items():
                    class_display = class_mapping.get(class_name, f"Class {class_name}")
                    st.write(f"- {class_display}: {len(crops)} detections")
                    if crops:
                        if len(crops) > 1:
                            student_crop_idx = st.selectbox(
                                f"Select {class_display}", 
                                range(len(crops)),
                                format_func=lambda x: f"Detection {x+1}",
                                key=f"student_{class_name}_selector"
                            )
                        else:
                            student_crop_idx = 0
                        
                        img = Image.open(crops[student_crop_idx])
                        st.image(img, caption=f"Student {class_display} ({student_crop_idx + 1}/{len(crops)})", width=350)
        
        progress_bar.progress(60)
        
        # Step 3: Similarity Scoring
        status_text.text("Calculating similarity scores...")
        st.subheader("🧮 Step 3: Similarity Scoring")
        
        with st.expander("View Scoring Process", expanded=True):
            scoring_results = scorer.calculate_scores(detection_results, manual_scores)
            progress_bar.progress(90)
            
            # Class mapping
            class_mapping = {
                "0": "📐 Formulas/Equations",
                "1": "📊 Figures/Diagrams", 
                "2": "📋 Tables/Structured Data"
            }
            
            # Display results for each class
            total_absolute_score = 0
            total_partial_score = 0
            all_individual_scores = []
            
            for class_name, class_results in scoring_results.items():
                class_display = class_mapping.get(class_name, f"Class {class_name}")
                st.write(f"**{class_display} Results:**")
                
                if 'matches' in class_results and class_results['matches']:
                    num_matches = len(class_results['matches'])
                    st.success(f"✅ Found {num_matches} comparisons - showing all below")
                    
                    # Display all comparisons in a single column for better visibility
                    for match_idx, match in enumerate(class_results['matches']):
                        st.subheader(f"Question {match_idx + 1}")
                        
                        # Create side-by-side layout for images
                        img_col1, img_col2 = st.columns(2)
                        
                        with img_col1:
                            ref_img = Image.open(match['reference'])
                            st.image(ref_img, caption="Reference", width=280)
                        
                        with img_col2:
                            student_img = Image.open(match['student'])
                            st.image(student_img, caption="Student", width=280)
                        
                        # Scores with pass/fail status
                        passed = match.get('passed_threshold', False)
                        status_color = "🟢" if passed else "🔴"
                        status_label = "PASS" if passed else "FAIL"
                        
                        # Calculate dual scores
                        question_weight = manual_scores[match_idx] if match_idx < len(manual_scores) else 3.0
                        absolute_score = question_weight if passed else 0.0
                        partial_score = question_weight * match['similarity']
                        
                        # Display metrics in columns
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric(
                                label="Similarity", 
                                value=f"{match['similarity']:.3f}",
                                delta=f"{status_color} {status_label}"
                            )
                        
                        with metric_col2:
                            st.metric(
                                label="Absolute Score", 
                                value=f"{absolute_score:.1f}",
                                delta="Boolean-based"
                            )
                        
                        with metric_col3:
                            st.metric(
                                label="Partial Score", 
                                value=f"{partial_score:.1f}",
                                delta="Similarity-based"
                            )
                        
                        # Show strategy's explanation
                        explanation = match.get('explanation', match.get('claude_explanation', 'No explanation available'))
                        if explanation and explanation != 'No explanation available':
                            strategy_emoji = "🤖" if strategy == "nova" else "📊"
                            strategy_name = "Nova" if strategy == "nova" else "CLIP"
                            with st.expander(f"{strategy_emoji} {strategy_name} Analysis", expanded=False):
                                st.write(explanation)
                        
                        st.divider()  # Add separator between comparisons
                        
                        # Accumulate totals for both scoring methods
                        total_absolute_score += absolute_score
                        total_partial_score += partial_score
                        
                else:
                    st.warning(f"⚠️ No matches found for {class_display}. This could mean no detections were found in one or both images.")
                
                # Calculate class totals for both methods
                class_absolute_total = sum(manual_scores[i] if match.get('passed_threshold', False) else 0.0 
                                         for i, match in enumerate(class_results.get('matches', [])))
                class_partial_total = sum(manual_scores[i] * match['similarity'] 
                                        for i, match in enumerate(class_results.get('matches', [])))
                
                all_individual_scores.extend(class_results.get('individual_scores', []))
                
                # Display class totals for both methods
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{class_display} Absolute Total: {class_absolute_total:.2f}**")
                with col2:
                    st.write(f"**{class_display} Partial Total: {class_partial_total:.2f}**")
                st.divider()
        
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        # Final Results
        st.header("🎉 Final Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Absolute Total Score",
                value=f"{total_absolute_score:.2f}",
                delta=f"Boolean-based scoring"
            )
        
        with col2:
            st.metric(
                label="Partial Total Score",
                value=f"{total_partial_score:.2f}",
                delta=f"Similarity-based scoring"
            )
        
        with col3:
            max_possible = sum(manual_scores)
            st.metric(
                label="Max Possible Score",
                value=f"{max_possible:.2f}",
                delta=f"{len(all_individual_scores)} questions"
            )
        
        # Show percentages
        st.subheader("📈 Score Percentages")
        perc_col1, perc_col2 = st.columns(2)
        
        with perc_col1:
            abs_percentage = (total_absolute_score / max_possible * 100) if max_possible > 0 else 0
            st.metric(
                label="Absolute Score %", 
                value=f"{abs_percentage:.1f}%"
            )
        
        with perc_col2:
            partial_percentage = (total_partial_score / max_possible * 100) if max_possible > 0 else 0
            st.metric(
                label="Partial Score %", 
                value=f"{partial_percentage:.1f}%"
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
            
            matches = class_results.get('matches', [])
            for i, (ind_score, sim_score) in enumerate(zip(individual_scores, similarity_scores)):
                # Get pass/fail status from match data
                if i < len(matches):
                    passed = matches[i].get('passed_threshold', False)
                    status = "🟢 PASS" if passed else "🔴 FAIL"
                else:
                    status = "🔴 FAIL"
                    passed = False
                
                # Calculate both scoring methods
                weight = manual_scores[min(question_num-1, len(manual_scores)-1)]
                absolute_score = weight if passed else 0.0
                partial_score = weight * sim_score
                
                detailed_results.append({
                    "Question": f"Q{question_num}",
                    "Type": class_display,
                    "Similarity": f"{sim_score:.4f}",
                    "Status": status,
                    "Weight": weight,
                    "Absolute Score": f"{absolute_score:.1f}",
                    "Partial Score": f"{partial_score:.1f}"
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