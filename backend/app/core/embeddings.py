import os
import json
import time
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.config import ROOT_DIR

class OptimizedEmbeddings:
    """Wrapper para Embeddings com Cache Simples e Rate Limit"""
    def __init__(self, model, use_cache=True, batch_size=10, delay=2.0):
        self.model = model
        self.use_cache = use_cache
        self.batch_size = batch_size
        self.delay = delay
        
        # Define cache em data/embeddings_cache
        self.cache_dir = os.path.join(ROOT_DIR, "data", "embeddings_cache")
        
        if use_cache and not os.path.exists(self.cache_dir):
            try: os.makedirs(self.cache_dir)
            except: pass 

    def _get_key(self, text):
        return hashlib.md5(text.encode()).hexdigest()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60), retry=retry_if_exception_type(Exception))
    def _embed_documents_with_retry(self, texts):
        return self.model.embed_documents(texts)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60), retry=retry_if_exception_type(Exception))
    def _embed_query_with_retry(self, text):
        return self.model.embed_query(text)

    def embed_documents(self, texts):
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            batch_result = []
            to_process = []
            indices = []

            for idx, text in enumerate(batch):
                key = self._get_key(text)
                path = os.path.join(self.cache_dir, f"{key}.json")
                if self.use_cache and os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            batch_result.append((idx, json.load(f)))
                    except:
                        to_process.append(text)
                        indices.append(idx)
                else:
                    to_process.append(text)
                    indices.append(idx)

            if to_process:
                try:
                    time.sleep(self.delay)
                    embeddings = self._embed_documents_with_retry(to_process)
                    for text, emb, idx in zip(to_process, embeddings, indices):
                        if self.use_cache:
                            try:
                                with open(os.path.join(self.cache_dir, f"{self._get_key(text)}.json"), 'w') as f:
                                    json.dump(emb, f)
                            except: pass
                        batch_result.append((idx, emb))
                except Exception as e:
                    print(f"Embedding Error: {e}")
                    raise e

            batch_result.sort(key=lambda x: x[0])
            results.extend([x[1] for x in batch_result])
        return results

    def embed_query(self, text):
        try:
            return self._embed_query_with_retry(text)
        except Exception as e:
            print(f"Query Embedding Failed: {e}")
            raise e