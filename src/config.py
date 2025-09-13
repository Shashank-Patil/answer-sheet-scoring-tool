"""
Configuration settings for the Answer Sheet Scoring Tool.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SimilarityConfig:
    """Configuration for similarity scoring strategies."""

    # AWS settings
    aws_region: str = "ap-south-1"

    # Model configurations
    nova_model_id: str = "apac.amazon.nova-lite-v1:0"
    clip_model_name: str = "openai/clip-vit-base-patch32"

    # Image processing
    max_image_size: int = 1024
    image_quality: int = 85

    # Caching
    enable_cache: bool = True
    cache_ttl_hours: int = 24

    # Scoring thresholds
    similarity_threshold: float = 0.85
    equivalent_threshold: float = 0.85

    # API settings
    max_tokens: int = 1000
    temperature: float = 0.1
    top_p: float = 0.9

    # Content type mappings
    content_type_descriptions: Dict[str, str] = None
    content_type_emojis: Dict[str, str] = None

    def __post_init__(self):
        if self.content_type_descriptions is None:
            self.content_type_descriptions = {
                "formula": "mathematical formulas or equations",
                "figure": "diagrams, charts, or figures",
                "table": "tables or structured data",
                "unknown": "content"
            }

        if self.content_type_emojis is None:
            self.content_type_emojis = {
                "formula": "📐 Formula",
                "figure": "📊 Figure",
                "table": "📋 Table"
            }


@dataclass
class DetectionConfig:
    """Configuration for object detection."""

    confidence_threshold: float = 0.25
    default_manual_scores: list = None

    def __post_init__(self):
        if self.default_manual_scores is None:
            self.default_manual_scores = [3.0] * 11


# Global configuration instance
CONFIG = SimilarityConfig()
DETECTION_CONFIG = DetectionConfig()