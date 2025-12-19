# 🔍 Análise: Rate Limit Gemini API

## 📊 Situação Atual

```
Total de requisições: 32/dia
Total de erros 429: 24/dia
Taxa de erro: 75%
```

**Problema crítico:** 3 em cada 4 requisições falham!

---

## 🎯 Causa Raiz

### 1. **Embeddings na Ingestão** (Principal Culpado)

Cada chunk de documento precisa de 1 chamada à API Gemini:

```python
# Situação atual na ingestão:
chunks = text_splitter.split_documents(documents)  # Ex: 100 chunks
embeddings = GoogleGenerativeAIEmbeddings(...)
vectorstore = Chroma.from_documents(chunks, embeddings)
# 👆 Isso faz 100 chamadas à API instantaneamente!
```

**Exemplo real:**
- 1 PDF de 50 páginas = ~50-100 chunks
- 1 URL = ~10-50 chunks
- **Total: 1 ingestão = 10-100 chamadas à API!**

### 2. **Chat** (Secundário)

Cada pergunta = 1 chamada à API Gemini

### 3. **Inicialização do RAG**

Ao carregar o vector store, pode tentar re-criar embeddings

---

## 📉 Limites da API Gemini (Free Tier)

```
Embeddings (text-embedding-004):
- 1,500 requisições/dia
- 15 requisições/minuto

Chat (gemini-2.0-flash-exp):
- 15 requisições/minuto
- 1,500 requisições/dia
```

**Com ingestão de 3 PDFs de 50 páginas:**
- 3 × 75 chunks = 225 requisições
- Em minutos: 225/15 = **15 minutos de espera!**
- Sem rate limiting: **Estoura o limite imediatamente**

---

## 💡 Soluções Propostas

### Solução 1: **Rate Limiting Inteligente** ⭐ (Recomendado)

Implementar controle de taxa com retry exponencial:

```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60)
)
def create_embeddings_with_retry(texts, embeddings_model):
    return embeddings_model.embed_documents(texts)
```

**Benefícios:**
- ✅ Respeita limites da API
- ✅ Retry automático em caso de erro
- ✅ Não perde dados
- ✅ Funciona com free tier

### Solução 2: **Batching com Delays**

Processar em lotes pequenos com pausas:

```python
def ingest_with_rate_limit(chunks, embeddings, batch_size=10):
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        vectorstore.add_documents(batch)
        
        # Pausa entre batches (15 req/min = 4s entre batches)
        if i + batch_size < len(chunks):
            time.sleep(5)  # 5 segundos entre batches
```

**Benefícios:**
- ✅ Controle fino da taxa
- ✅ Progresso visível
- ✅ Pode pausar/continuar

### Solução 3: **Cache de Embeddings** ⭐⭐ (Melhor!)

Salvar embeddings já calculados:

```python
import hashlib
import json

def get_embedding_with_cache(text, embeddings_model):
    # Hash do texto como chave
    cache_key = hashlib.md5(text.encode()).hexdigest()
    cache_file = f".embeddings_cache/{cache_key}.json"
    
    # Tentar carregar do cache
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    # Se não existe, calcular e salvar
    embedding = embeddings_model.embed_query(text)
    os.makedirs('.embeddings_cache', exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(embedding, f)
    
    return embedding
```

**Benefícios:**
- ✅ Re-ingestão do mesmo doc = 0 chamadas API
- ✅ Economia massiva
- ✅ Velocidade 100x maior em re-processamento

### Solução 4: **Limitar Chunks por Ingestão**

Adicionar limite configurável:

```python
MAX_CHUNKS_PER_INGESTION = int(os.getenv("MAX_CHUNKS_PER_INGESTION", "50"))

if len(chunks) > MAX_CHUNKS_PER_INGESTION:
    logger.warning(f"Too many chunks ({len(chunks)}), limiting to {MAX_CHUNKS_PER_INGESTION}")
    chunks = chunks[:MAX_CHUNKS_PER_INGESTION]
```

**Benefícios:**
- ✅ Previne estouros acidentais
- ✅ Protege a API key
- ✅ Feedback claro ao usuário

### Solução 5: **Embeddings Locais (Alternativa)**

Usar modelo local em vez de Gemini:

```python
from sentence_transformers import SentenceTransformer

# Modelo local (sem API)
local_model = SentenceTransformer('all-MiniLM-L6-v2')

class LocalEmbeddings:
    def embed_documents(self, texts):
        return local_model.encode(texts).tolist()
    
    def embed_query(self, text):
        return local_model.encode([text])[0].tolist()
```

**Benefícios:**
- ✅ Ilimitado (local)
- ✅ Gratuito
- ✅ Rápido
- ❌ Qualidade pode ser menor

---

## 🎯 Solução Recomendada (Híbrida)

Combinar múltiplas estratégias:

### 1. **Rate Limiting + Retry**
- Usar tenacity para retry automático
- Esperar entre requisições

### 2. **Batching Inteligente**
- Processar 10 chunks por vez
- Pausa de 5s entre batches
- Mostrar progresso ao usuário

### 3. **Cache de Embeddings**
- Salvar embeddings em disco
- Reutilizar em re-ingestão

### 4. **Limites Configuráveis**
- MAX_CHUNKS_PER_INGESTION=50
- BATCH_SIZE=10
- DELAY_BETWEEN_BATCHES=5

### 5. **Fallback para Local** (Opcional)
- Se rate limit persistir, usar modelo local
- Configurável via .env

---

## 📋 Implementação Proposta

### Arquivo: `backend/gemini_optimizer.py`

```python
import time
import os
import hashlib
import json
from typing import List
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

class GeminiRateLimitError(Exception):
    pass

class OptimizedEmbeddings:
    def __init__(self, embeddings_model, use_cache=True):
        self.model = embeddings_model
        self.use_cache = use_cache
        self.cache_dir = ".embeddings_cache"
        
        if use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def _load_from_cache(self, text: str):
        if not self.use_cache:
            return None
        
        cache_key = self._get_cache_key(text)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None
    
    def _save_to_cache(self, text: str, embedding):
        if not self.use_cache:
            return
        
        cache_key = self._get_cache_key(text)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        with open(cache_file, 'w') as f:
            json.dump(embedding, f)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(GeminiRateLimitError)
    )
    def _embed_with_retry(self, texts: List[str]):
        try:
            return self.model.embed_documents(texts)
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                logger.warning(f"Rate limit hit, will retry: {e}")
                raise GeminiRateLimitError(str(e))
            raise
    
    def embed_documents_batched(self, texts: List[str], batch_size=10, delay=5):
        """Embed documents in batches with rate limiting"""
        all_embeddings = []
        cache_hits = 0
        api_calls = 0
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = []
            texts_to_embed = []
            cache_indices = []
            
            # Check cache first
            for idx, text in enumerate(batch):
                cached = self._load_from_cache(text)
                if cached is not None:
                    batch_embeddings.append((idx, cached))
                    cache_hits += 1
                else:
                    texts_to_embed.append(text)
                    cache_indices.append(idx)
            
            # Embed remaining texts
            if texts_to_embed:
                logger.info(f"Embedding batch {i//batch_size + 1}, {len(texts_to_embed)} texts (cache hits: {cache_hits})")
                
                try:
                    new_embeddings = self._embed_with_retry(texts_to_embed)
                    api_calls += len(texts_to_embed)
                    
                    # Save to cache and add to results
                    for text, embedding, idx in zip(texts_to_embed, new_embeddings, cache_indices):
                        self._save_to_cache(text, embedding)
                        batch_embeddings.append((idx, embedding))
                    
                    # Rate limiting: wait between batches
                    if i + batch_size < len(texts):
                        logger.info(f"Waiting {delay}s before next batch...")
                        time.sleep(delay)
                        
                except Exception as e:
                    logger.error(f"Failed to embed batch: {e}")
                    raise
            
            # Sort by original index and add to results
            batch_embeddings.sort(key=lambda x: x[0])
            all_embeddings.extend([emb for _, emb in batch_embeddings])
        
        logger.info(f"Embedding complete: {cache_hits} cache hits, {api_calls} API calls")
        return all_embeddings
    
    def embed_query(self, text: str):
        """Embed single query (for search)"""
        # No cache for queries (they change frequently)
        return self.model.embed_query(text)
```

### Uso no `server.py`:

```python
from gemini_optimizer import OptimizedEmbeddings

# Na função de ingestão:
base_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=API_KEY
)

optimized_embeddings = OptimizedEmbeddings(
    base_embeddings,
    use_cache=True
)

# Extrair textos dos chunks
texts = [chunk.page_content for chunk in chunks]

# Embed com rate limiting
embeddings = optimized_embeddings.embed_documents_batched(
    texts,
    batch_size=10,
    delay=5
)

# Criar documentos com embeddings
# (código adicional necessário para integrar com ChromaDB)
```

---

## 📊 Impacto Esperado

### Antes (Situação Atual):
```
100 chunks = 100 chamadas API instantâneas
Tempo: <1 segundo
Erros 429: 75%
Taxa de sucesso: 25%
```

### Depois (Com Otimização):
```
100 chunks = 10 batches × 10 chunks
Tempo: ~50 segundos (10 batches × 5s)
Erros 429: <5%
Taxa de sucesso: >95%

Com cache (re-ingestão):
100 chunks = 0 chamadas API
Tempo: <1 segundo
Erros 429: 0%
Taxa de sucesso: 100%
```

---

## 🚀 Próximos Passos

### 1. Implementação Rápida (30 min)
- [ ] Criar `backend/gemini_optimizer.py`
- [ ] Adicionar tenacity ao requirements.txt
- [ ] Integrar no endpoint de ingestão PDF
- [ ] Integrar no endpoint de ingestão URL

### 2. Configuração (.env)
```env
# Rate Limiting
MAX_CHUNKS_PER_INGESTION=50
EMBEDDING_BATCH_SIZE=10
EMBEDDING_DELAY_SECONDS=5
USE_EMBEDDINGS_CACHE=true
```

### 3. Frontend (Feedback ao Usuário)
- Mostrar progresso: "Processando batch 3/10..."
- Estimativa de tempo
- Opção de cancelar

### 4. Monitoramento
- Log de API calls
- Contador de cache hits
- Alertas de rate limit

---

## 🎯 Resultado Final Esperado

```
✅ Taxa de erro: 75% → <5%
✅ Tempo de ingestão: Previsível (10s por 10 chunks)
✅ Re-ingestão: Instantânea (cache)
✅ Economia de API: 90% com cache
✅ Experiência do usuário: Transparente
```

---

**Implementar agora?** 🚀
