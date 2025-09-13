"""
Similarity scoring with two strategies: CLIP embeddings (default) and Nova vision model.
"""

import os
import base64
import json
from typing import List, Dict, Tuple
from PIL import Image
import io
from result_cache import ResultCache
from config import CONFIG

# CLIP Strategy imports
try:
    import torch
    import clip
    import torchvision.transforms as transforms
    CLIP_AVAILABLE = True
except ImportError:
    print("Warning: torch/clip not available for CLIP strategy. Install with: pip install torch clip-by-openai")
    CLIP_AVAILABLE = False

# Nova Strategy imports
try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    print("Warning: boto3 not available for Nova strategy. Install with: pip install boto3")
    BOTO3_AVAILABLE = False


class SimilarityScorer:
    """Similarity scorer with CLIP and Nova strategies."""

    def __init__(self, strategy: str = "clip", enable_cache: bool = True, cache_ttl_hours: int = 24):
        """
        Initialize similarity scorer.

        Args:
            strategy: 'clip' (default) or 'nova'
            enable_cache: Whether to enable result caching
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.strategy = strategy.lower()
        if self.strategy not in ["clip", "nova"]:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'clip' or 'nova'")

        # Initialize cache (only useful for Nova strategy with detailed explanations)
        self.enable_cache = enable_cache and (self.strategy == "nova")
        if self.enable_cache:
            self.cache = ResultCache(cache_ttl_hours=cache_ttl_hours)
            print(f"✓ Result caching enabled for Nova strategy (TTL: {cache_ttl_hours}h)")
        else:
            self.cache = None
            if enable_cache and self.strategy == "clip":
                print("ℹ️ Caching disabled for CLIP strategy (simple similarity scores don't benefit from caching)")

        # Strategy-specific initialization
        if self.strategy == "clip":
            self._init_clip()
        else:
            self._init_nova()

        print(f"✓ Initialized {self.strategy.upper()} similarity scorer")

    def _init_clip(self):
        """Initialize CLIP model."""
        if not CLIP_AVAILABLE:
            raise ImportError("torch and clip are required for CLIP strategy")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.image_transform = None
        self.model_id = "clip-vit-b-32"

    def _init_nova(self):
        """Initialize Nova client."""
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for Nova strategy")

        self.bedrock_client = None
        self.model_id = CONFIG.nova_model_id

    def _load_clip_model(self):
        """Load CLIP model if not already loaded."""
        if self.model is None:
            self.model, _ = clip.load("ViT-B/32", device=self.device)
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            print(f"Loaded CLIP model on device: {self.device}")

    def _init_bedrock_client(self):
        """Initialize AWS Bedrock client if not already initialized."""
        if self.bedrock_client is None:
            try:
                self.bedrock_client = boto3.client(
                    service_name='bedrock-runtime',
                    region_name=CONFIG.aws_region,
                    verify=False
                )
                print(f"Initialized AWS Bedrock client in region: {CONFIG.aws_region}")
            except Exception as e:
                print(f"Error initializing Bedrock client: {e}")
                raise

    def _calculate_clip_similarity(self, image1_path: str, image2_path: str, content_type: str = "unknown") -> Dict:
        """Calculate similarity using CLIP."""
        self._load_clip_model()

        try:
            # Load and preprocess images
            image1 = self.image_transform(Image.open(image1_path)).unsqueeze(0).to(self.device)
            image2 = self.image_transform(Image.open(image2_path)).unsqueeze(0).to(self.device)

            # Encode images
            with torch.no_grad():
                image1_encoding = self.model.encode_image(image1)
                image2_encoding = self.model.encode_image(image2)

            # Calculate cosine similarity
            similarity_score = torch.nn.functional.cosine_similarity(
                image1_encoding, image2_encoding
            ).item()

            # Normalize to 0-1 range
            similarity_score = (similarity_score + 1) / 2

            is_equivalent = similarity_score >= CONFIG.similarity_threshold

            return {
                "similarity_score": similarity_score,
                "explanation": f"CLIP similarity: {similarity_score:.4f}",
                "equivalent": is_equivalent
            }

        except Exception as e:
            print(f"Error in CLIP similarity calculation: {e}")
            return {
                "similarity_score": 0.0,
                "explanation": f"CLIP error: {str(e)}",
                "equivalent": False
            }

    def _encode_image_for_nova(self, image_path: str) -> str:
        """Encode image to base64 for Nova."""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                if max(img.width, img.height) > CONFIG.max_image_size:
                    img.thumbnail((CONFIG.max_image_size, CONFIG.max_image_size), Image.Resampling.LANCZOS)

                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=CONFIG.image_quality)
                img_bytes = buffer.getvalue()

                return base64.b64encode(img_bytes).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            raise

    def _calculate_nova_similarity(self, image1_path: str, image2_path: str, content_type: str = "unknown") -> Dict:
        """Calculate similarity using Nova."""
        self._init_bedrock_client()

        try:
            image1_b64 = self._encode_image_for_nova(image1_path)
            image2_b64 = self._encode_image_for_nova(image2_path)

            content_description = CONFIG.content_type_descriptions.get(content_type, "content")

            prompt = f"""I need you to compare these two images containing {content_description} and assess how similar they are.
Judge similarity along the following dimensions:

1. **Mathematical formulas**
    - Focus on the mathematical structure and operations.
    - Equivalent if the same variables, operators, and relationships are expressed, even if written in different styles (e.g., 1/x vs x⁻¹).
    - Different if there are missing terms, different operators, or a change in mathematical meaning (e.g., (a+b)² vs a²+b²).
    - Ignore handwriting style, formatting differences, or spacing.
2. **Figures/diagrams**
    - Focus on the elements shown (shapes, axes, connections, labels) and their relationships.
    - Equivalent if they represent the same relationships or structure, even if layout, style, or colors differ.
    - Different if elements are missing, added, rearranged in a way that changes meaning, or if values/relationships differ.
3. **Tables**
    - Focus on the rows, columns, headers, and cell values.
    - Equivalent if the same data is present, even if formatting or order differs.
    - Different if values change, rows/columns are added or missing, or headers differ in meaning.

### Scoring Guidelines

- **0.90–1.00** → Same meaning/content, differences only in style, formatting, or minor notation.
- **0.70–0.89** → Mostly similar, but some small differences in data, notation, or structure.
- **0.40–0.69** → Partially similar, with noticeable differences in content, values, or structure.
- **0.00–0.39** → Very different or unrelated.

Output JSON:
- "similarity_score": number between 0.0 and 1.0
- "explanation": concise reasoning
- "equivalent": true if score >= 0.85"""

            conversation = [{
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
            }]

            response = self.bedrock_client.converse(
                modelId=self.model_id,
                messages=conversation,
                inferenceConfig={
                    "maxTokens": CONFIG.max_tokens,
                    "temperature": CONFIG.temperature,
                    "topP": CONFIG.top_p
                }
            )

            nova_response = response["output"]["message"]["content"][0]["text"]

            try:
                if "```json" in nova_response:
                    json_str = nova_response.split("```json")[1].split("```")[0].strip()
                elif "```" in nova_response:
                    json_str = nova_response.split("```")[1].strip()
                else:
                    json_str = nova_response.strip()

                result = json.loads(json_str)

                if "similarity_score" not in result:
                    result["similarity_score"] = 0.5
                if "explanation" not in result:
                    result["explanation"] = "Unable to parse explanation"
                if "equivalent" not in result:
                    result["equivalent"] = result["similarity_score"] >= CONFIG.equivalent_threshold

                return result

            except json.JSONDecodeError:
                score = 0.5
                if "similarity_score" in nova_response:
                    try:
                        parts = nova_response.split("similarity_score")
                        if len(parts) > 1:
                            score_part = parts[1].split(",")[0].split("}")[0]
                            score = float(''.join(c for c in score_part if c.isdigit() or c == '.'))
                            score = max(0.0, min(1.0, score))
                    except:
                        pass

                return {
                    "similarity_score": score,
                    "explanation": nova_response,
                    "equivalent": score >= CONFIG.equivalent_threshold
                }

        except ClientError as e:
            print(f"AWS Bedrock error: {e}")
            return {
                "similarity_score": 0.0,
                "explanation": f"Nova error: {str(e)}",
                "equivalent": False
            }
        except Exception as e:
            print(f"Error in Nova similarity calculation: {e}")
            return {
                "similarity_score": 0.0,
                "explanation": f"Error: {str(e)}",
                "equivalent": False
            }

    def calculate_similarity_score(self, image1_path: str, image2_path: str, content_type: str = "unknown") -> Dict:
        """
        Calculate similarity between two images using the configured strategy.

        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            content_type: Type of content ("formula", "figure", "table", "unknown")

        Returns:
            Dictionary with similarity_score, explanation, and equivalent fields
        """
        # Check cache first
        if self.enable_cache and self.cache:
            cached_result = self.cache.get_cached_result(
                image1_path, image2_path, self.model_id, content_type
            )
            if cached_result:
                return cached_result

        # Calculate similarity using selected strategy
        if self.strategy == "clip":
            result = self._calculate_clip_similarity(image1_path, image2_path, content_type)
        else:
            result = self._calculate_nova_similarity(image1_path, image2_path, content_type)

        # Store result in cache
        if self.enable_cache and self.cache:
            self.cache.store_result(
                image1_path, image2_path, self.model_id, content_type, result
            )

        return result

    def find_best_match(self, reference_image: str, candidate_images: List[str], content_type: str = "unknown") -> Tuple[str, float, int, Dict]:
        """
        Find the best matching image from candidates.

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
            result = self.calculate_similarity_score(reference_image, candidate_image, content_type)

            score = result.get("similarity_score", 0.0)
            if score > best_score:
                best_score = score
                best_match = candidate_image
                best_index = i
                best_result = result

        return best_match, best_score, best_index, best_result

    def compare_image_sets(self, reference_dir: str, student_dir: str, content_type: str = "unknown",
                          manual_scores: List[float] = None) -> Dict:
        """
        Compare all images using the configured strategy.

        Args:
            reference_dir: Directory containing reference images
            student_dir: Directory containing student images
            content_type: Type of content ("formula", "figure", "table")
            manual_scores: List of manual scoring weights

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
            from config import DETECTION_CONFIG
            manual_scores = DETECTION_CONFIG.default_manual_scores[:len(reference_images)]
        elif len(manual_scores) < len(reference_images):
            manual_scores.extend([3.0] * (len(reference_images) - len(manual_scores)))

        results = {
            'individual_scores': [],
            'similarity_scores': [],
            'total_score': 0.0,
            'matches': [],
            'explanations': []
        }

        student_image_paths = [os.path.join(student_dir, img) for img in student_images]

        content_name = CONFIG.content_type_emojis.get(content_type, "Content")
        strategy_name = self.strategy.upper()
        print(f"\nUsing {strategy_name} to compare {content_name.lower()}s...")

        for i, ref_image in enumerate(reference_images):
            ref_path = os.path.join(reference_dir, ref_image)
            print(f"\nProcessing reference {content_name.lower()} {i+1}/{len(reference_images)}: {ref_image}")

            # Find best match
            best_match_path, max_similarity, match_index, best_result = self.find_best_match(
                ref_path, student_image_paths, content_type
            )

            if best_match_path:
                # Use strategy's decision
                is_equivalent = best_result.get('equivalent', False)
                if is_equivalent:
                    weighted_score = manual_scores[i]
                    score_status = "PASS"
                else:
                    weighted_score = 0.0
                    score_status = "FAIL"

                results['individual_scores'].append(weighted_score)
                results['similarity_scores'].append(max_similarity)
                results['explanations'].append(best_result.get('explanation', 'No explanation'))
                results['matches'].append({
                    'reference': ref_path,
                    'student': best_match_path,
                    'similarity': max_similarity,
                    'weighted_score': weighted_score,
                    'passed_threshold': is_equivalent,
                    'explanation': best_result.get('explanation', ''),
                    'equivalent': is_equivalent
                })

                explanation = best_result.get('explanation', '')
                print(f"✓ Best match similarity: {max_similarity:.4f} ({score_status})")
                print(f"  Score: {weighted_score:.2f}/{manual_scores[i]:.2f}")
                print(f"  {strategy_name}: {explanation}")

            else:
                results['individual_scores'].append(0.0)
                results['similarity_scores'].append(0.0)
                results['explanations'].append("No match found")
                print(f"✗ No match found for reference: {ref_image}")

        results['total_score'] = sum(results['individual_scores'])

        print(f"\n{content_name} Results Summary:")
        print(f"Individual Scores: {results['individual_scores']}")
        print(f"Total Score: {results['total_score']:.2f}")

        return results

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        if self.enable_cache and self.cache:
            return self.cache.get_cache_stats()
        return {"message": "Caching is disabled"}

    def print_cache_stats(self):
        """Print cache statistics."""
        if self.enable_cache and self.cache:
            self.cache.print_cache_stats()
        else:
            print("📊 Caching is disabled")

    def clear_cache(self):
        """Clear all cached results."""
        if self.enable_cache and self.cache:
            self.cache.clear_cache()
        else:
            print("Caching is disabled")