"""
Gemini API Rate Limiting and Optimization
Implements caching, batching, and retry logic to avoid 429 errors
"""

import time
import os
import hashlib
import json
from typing import List, Optional
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

class GeminiRateLimitError(Exception):
    """Custom exception for rate limit errors"""
    pass


class OptimizedEmbeddings:
    """
    Wrapper for GoogleGenerativeAIEmbeddings with:
    - Caching (avoid re-computing same embeddings)
    - Batching (process in small groups)
    - Rate limiting (respect API limits)
    - Retry logic (handle transient errors)
    """
    
    def __init__(
        self,
        embeddings_model,
        use_cache: bool = True,
        batch_size: int = 10,
        delay_between_batches: float = 5.0,
        cache_dir: str = ".embeddings_cache"
    ):
        self.model = embeddings_model
        self.use_cache = use_cache
        self.batch_size = batch_size
        self.delay = delay_between_batches
        self.cache_dir = cache_dir
        
        if use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Embeddings cache enabled at {cache_dir}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text content"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _load_from_cache(self, text: str) -> Optional[List[float]]:
        """Load embedding from cache if exists"""
        if not self.use_cache:
            return None
        
        cache_key = self._get_cache_key(text)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache for {cache_key}: {e}")
                return None
        
        return None
    
    def _save_to_cache(self, text: str, embedding: List[float]):
        """Save embedding to cache"""
        if not self.use_cache:
            return
        
        cache_key = self._get_cache_key(text)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to save cache for {cache_key}: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(GeminiRateLimitError)
    )
    def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Embed texts with automatic retry on rate limit"""
        try:
            return self.model.embed_documents(texts)
        except Exception as e:
            error_str = str(e)
            # Check if it's a rate limit error
            if "429" in error_str or "rate limit" in error_str.lower() or "quota" in error_str.lower():
                logger.warning(f"Rate limit hit, will retry: {error_str[:200]}")
                raise GeminiRateLimitError(error_str)
            # Re-raise other errors
            logger.error(f"Embedding error: {error_str[:200]}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents with batching, caching, and rate limiting
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        total_texts = len(texts)
        all_embeddings = []
        cache_hits = 0
        api_calls = 0
        
        logger.info(f"Starting to embed {total_texts} texts")
        
        # Process in batches
        for batch_idx in range(0, total_texts, self.batch_size):
            batch_end = min(batch_idx + self.batch_size, total_texts)
            batch = texts[batch_idx:batch_end]
            batch_num = (batch_idx // self.batch_size) + 1
            total_batches = (total_texts + self.batch_size - 1) // self.batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
            
            # Check cache for each text in batch
            batch_results = []
            texts_to_embed = []
            text_indices = []
            
            for idx, text in enumerate(batch):
                cached = self._load_from_cache(text)
                if cached is not None:
                    batch_results.append((batch_idx + idx, cached))
                    cache_hits += 1
                else:
                    texts_to_embed.append(text)
                    text_indices.append(batch_idx + idx)
            
            # Embed texts not in cache
            if texts_to_embed:
                try:
                    new_embeddings = self._embed_with_retry(texts_to_embed)
                    api_calls += len(texts_to_embed)
                    
                    # Save to cache and add to results
                    for text, embedding, global_idx in zip(texts_to_embed, new_embeddings, text_indices):
                        self._save_to_cache(text, embedding)
                        batch_results.append((global_idx, embedding))
                    
                    logger.info(f"Batch {batch_num}: {len(texts_to_embed)} API calls, {len(batch) - len(texts_to_embed)} cache hits")
                    
                    # Rate limiting: wait between batches (except last one)
                    if batch_end < total_texts:
                        logger.info(f"Waiting {self.delay}s before next batch...")
                        time.sleep(self.delay)
                        
                except Exception as e:
                    logger.error(f"Failed to embed batch {batch_num}: {e}")
                    # Re-raise to stop processing
                    raise
            else:
                logger.info(f"Batch {batch_num}: All {len(batch)} from cache!")
            
            # Sort by original index
            batch_results.sort(key=lambda x: x[0])
            all_embeddings.extend([emb for _, emb in batch_results])
        
        # Log statistics
        cache_rate = (cache_hits / total_texts * 100) if total_texts > 0 else 0
        logger.info(
            f"Embedding complete: {total_texts} texts, "
            f"{cache_hits} cache hits ({cache_rate:.1f}%), "
            f"{api_calls} API calls"
        )
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text (for search)
        Note: Queries are NOT cached as they're typically unique
        """
        try:
            return self.model.embed_query(text)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate limit" in error_str.lower():
                logger.warning("Rate limit on query embedding")
                # Wait and retry once
                time.sleep(10)
                return self.model.embed_query(text)
            raise


def get_embedding_stats(cache_dir: str = ".embeddings_cache") -> dict:
    """Get statistics about embeddings cache"""
    if not os.path.exists(cache_dir):
        return {"cached_embeddings": 0, "cache_size_mb": 0}
    
    files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
    
    total_size = 0
    for f in files:
        total_size += os.path.getsize(os.path.join(cache_dir, f))
    
    return {
        "cached_embeddings": len(files),
        "cache_size_mb": round(total_size / (1024 * 1024), 2)
    }


def clear_embedding_cache(cache_dir: str = ".embeddings_cache"):
    """Clear all cached embeddings"""
    if not os.path.exists(cache_dir):
        return
    
    import shutil
    shutil.rmtree(cache_dir)
    logger.info(f"Cleared embeddings cache at {cache_dir}")
