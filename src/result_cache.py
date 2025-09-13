"""
Result caching system for similarity scoring to avoid redundant API calls.
Stores results based on image hashes and model configurations.
"""

import os
import json
import hashlib
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path


class ResultCache:
    """Handles caching of similarity scoring results."""
    
    def __init__(self, cache_dir: str = "cache", cache_ttl_hours: int = 24):
        """
        Initialize result cache.
        
        Args:
            cache_dir: Directory to store cache files
            cache_ttl_hours: Time-to-live for cached results in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.cache_file = self.cache_dir / "similarity_cache.json"
        self.cache_data = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load existing cache data from file."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    # Clean expired entries
                    return self._clean_expired_entries(cache_data)
            return {}
        except Exception as e:
            print(f"Error loading cache: {e}")
            return {}
    
    def _save_cache(self):
        """Save cache data to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache_data, f, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def _clean_expired_entries(self, cache_data: Dict) -> Dict:
        """Remove expired entries from cache."""
        current_time = datetime.now()
        cleaned_data = {}
        
        for key, entry in cache_data.items():
            try:
                entry_time = datetime.fromisoformat(entry.get('timestamp', ''))
                if current_time - entry_time < self.cache_ttl:
                    cleaned_data[key] = entry
                else:
                    print(f"Removing expired cache entry: {key}")
            except Exception:
                # Skip malformed entries
                continue
        
        return cleaned_data
    
    def _get_image_hash(self, image_path: str) -> str:
        """Calculate hash of image file for cache key."""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                return hashlib.md5(image_data).hexdigest()
        except Exception as e:
            print(f"Error hashing image {image_path}: {e}")
            return f"error_{os.path.basename(image_path)}"
    
    def _get_cache_key(self, image1_path: str, image2_path: str, 
                      model_id: str, content_type: str) -> str:
        """Generate cache key for the comparison."""
        hash1 = self._get_image_hash(image1_path)
        hash2 = self._get_image_hash(image2_path)
        
        # Create deterministic key regardless of image order
        if hash1 < hash2:
            key_base = f"{hash1}_{hash2}"
        else:
            key_base = f"{hash2}_{hash1}"
        
        # Include model and content type in key
        cache_key = f"{key_base}_{model_id}_{content_type}"
        return hashlib.md5(cache_key.encode()).hexdigest()
    
    def get_cached_result(self, image1_path: str, image2_path: str, 
                         model_id: str, content_type: str = "unknown") -> Optional[Dict]:
        """
        Get cached similarity result if available.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            model_id: Model identifier used for comparison
            content_type: Type of content being compared
            
        Returns:
            Cached result dictionary or None if not found
        """
        cache_key = self._get_cache_key(image1_path, image2_path, model_id, content_type)
        
        if cache_key in self.cache_data:
            entry = self.cache_data[cache_key]
            print(f"✓ Using cached result for {os.path.basename(image1_path)} vs {os.path.basename(image2_path)}")
            return entry['result']
        
        return None
    
    def store_result(self, image1_path: str, image2_path: str, model_id: str, 
                    content_type: str, result: Dict):
        """
        Store similarity result in cache.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            model_id: Model identifier used for comparison
            content_type: Type of content being compared
            result: Result dictionary to cache
        """
        cache_key = self._get_cache_key(image1_path, image2_path, model_id, content_type)
        
        self.cache_data[cache_key] = {
            'result': result,
            'timestamp': datetime.now().isoformat(),
            'image1': os.path.basename(image1_path),
            'image2': os.path.basename(image2_path),
            'model_id': model_id,
            'content_type': content_type
        }
        
        self._save_cache()
        print(f"✓ Cached result for {os.path.basename(image1_path)} vs {os.path.basename(image2_path)}")
    
    def clear_cache(self):
        """Clear all cached results."""
        self.cache_data = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        print("✓ Cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        total_entries = len(self.cache_data)
        
        # Count by content type
        content_type_counts = {}
        model_counts = {}
        
        for entry in self.cache_data.values():
            content_type = entry.get('content_type', 'unknown')
            model_id = entry.get('model_id', 'unknown')
            
            content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
            model_counts[model_id] = model_counts.get(model_id, 0) + 1
        
        return {
            'total_entries': total_entries,
            'content_type_breakdown': content_type_counts,
            'model_breakdown': model_counts,
            'cache_size_mb': self._get_cache_size_mb()
        }
    
    def _get_cache_size_mb(self) -> float:
        """Get cache file size in MB."""
        try:
            if self.cache_file.exists():
                size_bytes = self.cache_file.stat().st_size
                return round(size_bytes / (1024 * 1024), 2)
            return 0.0
        except Exception:
            return 0.0
    
    def print_cache_stats(self):
        """Print cache statistics to console."""
        stats = self.get_cache_stats()
        
        print("\n" + "="*50)
        print("📊 CACHE STATISTICS")
        print("="*50)
        print(f"Total cached results: {stats['total_entries']}")
        print(f"Cache file size: {stats['cache_size_mb']} MB")
        
        if stats['content_type_breakdown']:
            print("\nBy content type:")
            for content_type, count in stats['content_type_breakdown'].items():
                print(f"  - {content_type}: {count}")
        
        if stats['model_breakdown']:
            print("\nBy model:")
            for model_id, count in stats['model_breakdown'].items():
                model_name = model_id.split('.')[-1] if '.' in model_id else model_id
                print(f"  - {model_name}: {count}")
        
        print("="*50)