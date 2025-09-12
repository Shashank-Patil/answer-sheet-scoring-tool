"""
Claude-based similarity scoring using AWS Bedrock.
Uses Claude Sonnet's vision capabilities to compare answer images.
"""

import os
import base64
import json
from typing import List, Dict, Tuple
from PIL import Image
import io

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    print("Warning: boto3 not available. Please install with: pip install boto3")
    BOTO3_AVAILABLE = False


class ClaudeSimilarityScorer:
    """Uses Claude Sonnet via AWS Bedrock for intelligent image similarity comparison."""
    
    def __init__(self, aws_region: str = "ap-south-1"):
        """
        Initialize Claude similarity scorer.
        
        Args:
            aws_region: AWS region for Bedrock service
        """
        self.aws_region = aws_region
        self.bedrock_client = None
        self.model_id = "apac.amazon.nova-lite-v1:0"  # Latest Claude Sonnet
        
    def initialize_bedrock(self):
        """Initialize AWS Bedrock client."""
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for Claude similarity scoring. Install with: pip install boto3")
        
        if self.bedrock_client is None:
            try:
                self.bedrock_client = boto3.client(
                    service_name='bedrock-runtime',
                    region_name=self.aws_region,
                    verify=False
                )
                print(f"Initialized AWS Bedrock client in region: {self.aws_region}")
            except Exception as e:
                print(f"Error initializing Bedrock client: {e}")
                print("Make sure AWS credentials are configured and you have Bedrock access.")
                raise
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image to base64 for Claude API.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large (Claude has size limits)
                max_size = 1024
                if max(img.width, img.height) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Convert to bytes
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                img_bytes = buffer.getvalue()
                
                # Encode to base64
                return base64.b64encode(img_bytes).decode('utf-8')
                
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            raise
    
    def calculate_similarity_with_claude(self, image1_path: str, image2_path: str, content_type: str = "unknown") -> Dict:
        """
        Use Claude to compare two images and determine similarity.
        
        Args:
            image1_path: Path to first image (reference)
            image2_path: Path to second image (student)
            content_type: Type of content ("formula", "figure", "table", or "unknown")
            
        Returns:
            Dictionary with similarity score and explanation
        """
        self.initialize_bedrock()
        
        try:
            # Encode both images
            image1_b64 = self.encode_image_to_base64(image1_path)
            image2_b64 = self.encode_image_to_base64(image2_path)
            
            # Create content type specific prompt
            type_prompts = {
                "formula": "mathematical formulas or equations",
                "figure": "diagrams, charts, or figures", 
                "table": "tables or structured data",
                "unknown": "content"
            }
            content_description = type_prompts.get(content_type, "content")
            
            # Construct the prompt
            prompt = f"""I need you to compare these two images containing {content_description} and determine how similar they are.

Please analyze:
1. The overall structure and layout
2. The specific content (mathematical expressions, visual elements, data, etc.)
3. The accuracy of the representation

For mathematical formulas: Focus on whether the mathematical expressions are equivalent, even if handwriting styles differ.
For figures/diagrams: Compare the structure, elements, and relationships shown.
For tables: Compare data organization, values, and structure.

Return your response as a JSON object with:
- "similarity_score": A number between 0.0 and 1.0 (where 1.0 means identical/equivalent content)
- "explanation": A brief explanation of your assessment
- "key_differences": List of main differences found (if any)
- "equivalent": Boolean indicating if the content is functionally equivalent

Be strict in your evaluation - only give high scores (>0.85) if the content is truly equivalent or nearly identical."""

            # Prepare the conversation for Amazon Nova
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"text": prompt},
                        {
                            "image": {
                                "format": "jpeg",
                                "source": {"bytes": base64.b64decode(image1_b64)}
                            }
                        },
                        {
                            "image": {
                                "format": "jpeg", 
                                "source": {"bytes": base64.b64decode(image2_b64)}
                            }
                        }
                    ]
                }
            ]
            
            # Call Amazon Nova via Bedrock converse API
            response = self.bedrock_client.converse(
                modelId=self.model_id,
                messages=conversation,
                inferenceConfig={
                    "maxTokens": 1000,
                    "temperature": 0.1,
                    "topP": 0.9
                }
            )
            
            # Parse response for Amazon Nova converse API
            claude_response = response["output"]["message"]["content"][0]["text"]
            
            # Try to parse JSON from Claude's response
            try:
                # Claude might wrap JSON in code blocks
                if "```json" in claude_response:
                    json_str = claude_response.split("```json")[1].split("```")[0].strip()
                elif "```" in claude_response:
                    json_str = claude_response.split("```")[1].strip()
                else:
                    json_str = claude_response.strip()
                
                result = json.loads(json_str)
                
                # Ensure required fields exist
                if "similarity_score" not in result:
                    result["similarity_score"] = 0.5
                if "explanation" not in result:
                    result["explanation"] = "Unable to parse explanation"
                if "equivalent" not in result:
                    result["equivalent"] = result["similarity_score"] >= 0.85
                
                return result
                
            except json.JSONDecodeError:
                print(f"Could not parse JSON from Claude response: {claude_response}")
                # Fallback: try to extract similarity score from text
                score = 0.5
                if "similarity_score" in claude_response:
                    try:
                        # Simple regex-like extraction
                        parts = claude_response.split("similarity_score")
                        if len(parts) > 1:
                            score_part = parts[1].split(",")[0].split("}")[0]
                            score = float(''.join(c for c in score_part if c.isdigit() or c == '.'))
                            score = max(0.0, min(1.0, score))
                    except:
                        pass
                
                return {
                    "similarity_score": score,
                    "explanation": claude_response[:200] + "...",
                    "equivalent": score >= 0.85,
                    "raw_response": claude_response
                }
                
        except ClientError as e:
            print(f"AWS Bedrock error: {e}")
            return {
                "similarity_score": 0.0,
                "explanation": f"Error calling Claude: {str(e)}",
                "equivalent": False
            }
        except Exception as e:
            print(f"Error in Claude similarity calculation: {e}")
            return {
                "similarity_score": 0.0,
                "explanation": f"Error: {str(e)}",
                "equivalent": False
            }
    
    def find_best_match(self, reference_image: str, candidate_images: List[str], content_type: str = "unknown") -> Tuple[str, float, int, Dict]:
        """
        Find the best matching image using Claude comparison.
        
        Args:
            reference_image: Path to reference image
            candidate_images: List of candidate image paths
            content_type: Type of content being compared
            
        Returns:
            Tuple of (best_match_path, similarity_score, index, full_result)
        """
        if not candidate_images:
            return None, 0.0, -1, {}
        
        best_match = None
        best_score = 0.0
        best_index = -1
        best_result = {}
        
        for i, candidate_image in enumerate(candidate_images):
            print(f"Comparing with candidate {i+1}/{len(candidate_images)}...")
            result = self.calculate_similarity_with_claude(reference_image, candidate_image, content_type)
            
            score = result.get("similarity_score", 0.0)
            if score > best_score:
                best_score = score
                best_match = candidate_image
                best_index = i
                best_result = result
        
        return best_match, best_score, best_index, best_result
    
    def compare_image_sets(self, reference_dir: str, student_dir: str, content_type: str = "unknown",
                          manual_scores: List[float] = None,
                          similarity_threshold: float = 0.85) -> Dict:
        """
        Compare all images using Claude's vision capabilities.
        
        Args:
            reference_dir: Directory containing reference images
            student_dir: Directory containing student images
            content_type: Type of content ("formula", "figure", "table")
            manual_scores: List of manual scoring weights
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
            print("No images found in one or both directories")
            return {}
        
        # Default manual scores
        if manual_scores is None:
            manual_scores = [3.0] * len(reference_images)
        elif len(manual_scores) < len(reference_images):
            manual_scores.extend([3.0] * (len(reference_images) - len(manual_scores)))
        
        results = {
            'individual_scores': [],
            'similarity_scores': [],
            'total_score': 0.0,
            'matches': [],
            'claude_explanations': []
        }
        
        student_image_paths = [os.path.join(student_dir, img) for img in student_images]
        
        content_names = {"formula": "📐 Formula", "figure": "📊 Figure", "table": "📋 Table"}
        content_name = content_names.get(content_type, f"Content")
        print(f"\nUsing Claude Sonnet to compare {content_name.lower()}s...")
        
        for i, ref_image in enumerate(reference_images):
            ref_path = os.path.join(reference_dir, ref_image)
            print(f"\nProcessing reference {content_name.lower()} {i+1}/{len(reference_images)}: {ref_image}")
            
            # Find best match using Claude
            best_match_path, max_similarity, match_index, claude_result = self.find_best_match(
                ref_path, student_image_paths, content_type
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
                results['claude_explanations'].append(claude_result.get('explanation', 'No explanation'))
                results['matches'].append({
                    'reference': ref_path,
                    'student': best_match_path,
                    'similarity': max_similarity,
                    'weighted_score': weighted_score,
                    'passed_threshold': max_similarity >= similarity_threshold,
                    'claude_explanation': claude_result.get('explanation', ''),
                    'equivalent': claude_result.get('equivalent', False)
                })
                
                explanation_preview = claude_result.get('explanation', '')[:100]
                print(f"✓ Best match similarity: {max_similarity:.4f} ({score_status})")
                print(f"  Score: {weighted_score:.2f}/{manual_scores[i]:.2f}")
                print(f"  Claude says: {explanation_preview}...")
                
            else:
                results['individual_scores'].append(0.0)
                results['similarity_scores'].append(0.0)
                results['claude_explanations'].append("No match found")
                print(f"✗ No match found for reference: {ref_image}")
        
        results['total_score'] = sum(results['individual_scores'])
        
        print(f"\n{content_name} Results Summary:")
        print(f"Individual Scores: {results['individual_scores']}")
        print(f"Total Score: {results['total_score']:.2f}")
        
        return results