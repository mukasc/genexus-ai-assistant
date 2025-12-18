import os
import sys
import logging
import json
import time
import hashlib
import tempfile
import shutil
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

# Third-party imports
from pythonjsonlogger import jsonlogger
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. CARREGAMENTO DE AMBIENTE ---
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# ==============================================================================
# 2. OTIMIZADOR GEMINI (INTEGRADO PARA EVITAR ERRO 502)
# ==============================================================================
class OptimizedEmbeddings:
    """Wrapper para Embeddings com Cache Simples e Rate Limit"""
    def __init__(self, model, use_cache=True, batch_size=10, delay=2.0):
        self.model = model
        self.use_cache = use_cache
        self.batch_size = batch_size
        self.delay = delay
        self.cache_dir = ".embeddings_cache"
        if use_cache and not os.path.exists(self.cache_dir):
            try: os.makedirs(self.cache_dir)
            except: pass 

    def _get_key(self, text):
        return hashlib.md5(text.encode()).hexdigest()

    # --- MÉTODOS COM RETRY ADICIONADOS ---
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
                    # ALTERADO: Usa o método protegido com retry
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
        # ALTERADO: Usa o método protegido com retry
        try:
            return self._embed_query_with_retry(text)
        except Exception as e:
            print(f"Query Embedding Failed: {e}")
            raise e

# ==============================================================================
# 3. CONFIGURAÇÃO WHITE LABEL (SAFE LOAD)
# ==============================================================================
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_config.json")

DEFAULT_CONFIG = {
    "identity": {
        "app_name": "AI Assistant (Default)",
        "app_subtitle": "System running in default mode",
        "welcome_message": "Config file not found.",
        "primary_color": "#333333",
        "secondary_color": "#555555",
        "logo_emoji": "⚠️"
    },
    "llm": {
        "model_name": "gemini-2.5-flash",
        "temperature": 0.1,
        "system_prompt": "You are a helpful assistant. Context: {context} Question: {question}"
    },
    "storage": {
        "collection_name": "default_collection",
        "persist_directory": "./chroma_db"
    },
    "ingestion": {
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
}

def load_app_config() -> Dict[str, Any]:
    """Carrega configuração do JSON ou usa Default se falhar"""
    if not os.path.exists(CONFIG_FILE):
        print(f"⚠️ AVISO: Arquivo {CONFIG_FILE} não encontrado. Usando Default.")
        return DEFAULT_CONFIG
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ ERRO: Falha ao ler JSON ({e}). Usando Default.")
        return DEFAULT_CONFIG

APP_CONFIG = load_app_config()

# ==============================================================================
# 4. LOGGING SETUP
# ==============================================================================
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH")

def resolve_log_file_path() -> str:
    if LOG_FILE_PATH: return LOG_FILE_PATH
    
    # Tenta caminho local
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend.out.log")
    
    # Cria arquivo se não existir
    if not os.path.exists(local_path):
        try:
            with open(local_path, 'a') as f: pass
        except: pass
    return local_path

CURRENT_LOG_FILE = resolve_log_file_path()

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        if not log_record.get('timestamp'):
            now = datetime.fromtimestamp(record.created) if hasattr(record, 'created') else datetime.now()
            log_record['timestamp'] = now.strftime('%Y-%m-%dT%H:%M:%S')
        if not log_record.get('level'):
            log_record['level'] = record.levelname.upper() if record.levelname else 'INFO'

def setup_logging():
    log_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(CURRENT_LOG_FILE)
    
    formatter = CustomJsonFormatter(fmt='%(timestamp)s %(level)s %(name)s %(message)s %(module)s %(funcName)s %(lineno)d')
    
    log_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Logger da Aplicação (Escreve APENAS no arquivo para evitar duplicidade)
    app_logger = logging.getLogger(__name__)
    app_logger.setLevel(logging.INFO)
    app_logger.handlers = [file_handler]
    app_logger.propagate = False 
    
    # Intercepta bibliotecas
    for lib in ['uvicorn', 'uvicorn.access', 'uvicorn.error', 'fastapi']:
        l = logging.getLogger(lib)
        l.handlers = [file_handler]
        l.propagate = False
    
    # Root Logger (Safety net para console)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [log_handler]
    
    return app_logger

logger = setup_logging()

# ==============================================================================
# 5. VARIÁVEIS GLOBAIS E LÓGICA RAG
# ==============================================================================
API_KEY = os.getenv("GEMINI_API_KEY")
rag_chain = None
vectorstore_instance = None 

def get_optimized_embeddings():
    if not API_KEY: raise ValueError("GEMINI_API_KEY missing")
    base = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=API_KEY, 
        transport="rest", 
        task_type="retrieval_document"
    )
    # Usa a classe interna (sem import externo)
    return OptimizedEmbeddings(base, use_cache=True, batch_size=10, delay=2.0)

def get_vectorstore():
    coll_name = APP_CONFIG.get('storage', {}).get('collection_name', 'default')
    p_dir = APP_CONFIG.get('storage', {}).get('persist_directory', './chroma_db')
    abs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), p_dir)
    
    return Chroma(
        collection_name=coll_name, 
        persist_directory=abs_dir, 
        embedding_function=get_optimized_embeddings()
    )

def initialize_rag_system():
    global rag_chain, vectorstore_instance
    if not API_KEY: return {"success": False, "error": "No API Key"}

    try:
        vectorstore_instance = get_vectorstore()
        retriever = vectorstore_instance.as_retriever(search_kwargs={"k": 3})
        
        # Configs do JSON
        model = APP_CONFIG.get('llm', {}).get('model_name', 'gemini-2.5-flash')
        temp = APP_CONFIG.get('llm', {}).get('temperature', 0.1)
        sys_prompt = APP_CONFIG.get('llm', {}).get('system_prompt', "Context: {context} Question: {question}")
        
        llm = ChatGoogleGenerativeAI(
            model=model, 
            temperature=temp, 
            google_api_key=API_KEY, 
            max_retries=1, 
            transport="rest"
        )
        
        prompt = ChatPromptTemplate.from_template(sys_prompt)
        
        def format_docs(docs):
            # Log de diagnóstico: Tamanho do contexto
            content = "\n\n".join(d.page_content for d in docs)
            if docs:
                logger.info(f"Retrieved {len(docs)} docs, total chars: {len(content)}", extra={"docs_found": True})
            else:
                logger.warning("No docs found for query", extra={"docs_found": False})
            return content if docs else "No context."
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return {"success": True, "message": f"Initialized '{APP_CONFIG.get('identity', {}).get('app_name')}'"}
    except Exception as e:
        logger.error(f"RAG Init Error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# --- RETRY AJUSTADO PARA EVITAR FLOODING ---
@retry(
    stop=stop_after_attempt(3), # Tenta 3x (Total)
    wait=wait_exponential(multiplier=2, min=5, max=30), # Espera 5s, 10s, 20s
    retry=retry_if_exception_type(Exception)
)
def run_chain_with_retry(chain, message):
    #logger.info("Invoking chain...") # Removido para não poluir, logamos o resultado depois
    return chain.invoke(message)

# ==============================================================================
# 6. APP E ENDPOINTS
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Startup: {APP_CONFIG.get('identity', {}).get('app_name')}")
    initialize_rag_system()
    yield
    logger.info("Shutdown")

app = FastAPI(title=APP_CONFIG.get('identity', {}).get('app_name'), lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatRequest(BaseModel): message: str
class ChatResponse(BaseModel): response: str; context_used: bool; error: Optional[str] = None; retry_after: Optional[int] = None
class IngestionResponse(BaseModel): status: str; message: str; chunks_created: Optional[int] = None
class HealthResponse(BaseModel): status: str; api_key_configured: bool; database_loaded: bool; message: str; app_name: str
class IndexStatusResponse(BaseModel): exists: bool; document_count: int; collection_name: str; message: str

# Endpoints
@app.get("/api/config")
async def get_frontend_config():
    """Retorna configurações visuais para o Frontend"""
    return APP_CONFIG.get('identity', DEFAULT_CONFIG['identity'])

@app.get("/api/logs")
async def get_logs(lines: int = 100, level: Optional[str] = None, search: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None):
    if not os.path.exists(CURRENT_LOG_FILE): return {"logs": [], "error": "Log file missing"}
    
    try:
        dt_start, dt_end = None, None
        log_fmt = '%Y-%m-%dT%H:%M:%S'
        
        if start_time:
            try: dt_start = datetime.strptime(start_time.replace(' ', 'T').split('.')[0].replace('Z',''), log_fmt)
            except: pass
        if end_time:
            try: dt_end = datetime.strptime(end_time.replace(' ', 'T').split('.')[0].replace('Z',''), log_fmt)
            except: pass
            
        filtered = []
        with open(CURRENT_LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    entry = json.loads(line)
                    # Filtros
                    lvl = entry.get('level') or ''
                    if level and lvl.upper() != level.upper(): continue
                    if search and search.lower() not in str(entry).lower(): continue
                    
                    if dt_start or dt_end:
                        ts = entry.get('timestamp')
                        if not ts: continue
                        try:
                            ldt = datetime.strptime(ts, log_fmt)
                            if dt_start and ldt < dt_start: continue
                            if dt_end and ldt > dt_end: continue
                        except: continue
                    filtered.append(entry)
                except: pass
        return {"logs": filtered[-lines:], "total_matches": len(filtered), "source": CURRENT_LOG_FILE}
    except Exception as e:
        logger.error(f"Log Error: {e}")
        return {"logs": [], "error": str(e)}

@app.get("/api/ping-ai")
async def ping_ai():
    if not API_KEY: raise HTTPException(status_code=500, detail="No API Key")
    try:
        model = APP_CONFIG.get('llm', {}).get('model_name', 'gemini-2.5-flash')
        llm = ChatGoogleGenerativeAI(model=model, google_api_key=API_KEY, temperature=0, max_retries=1)
        res = llm.invoke("Pong")
        return {"status": "success", "reply": res.content, "model": model}
    except Exception as e: return {"status": "error", "detail": str(e)}

@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if rag_chain else "degraded",
        api_key_configured=bool(API_KEY),
        database_loaded=rag_chain is not None,
        message="OK",
        app_name=APP_CONFIG.get('identity', {}).get('app_name', 'Unknown')
    )

@app.get("/api/index-status", response_model=IndexStatusResponse)
async def index_status():
    try:
        vs = vectorstore_instance or get_vectorstore()
        return IndexStatusResponse(exists=True, document_count=vs._collection.count(), collection_name=APP_CONFIG.get('storage',{}).get('collection_name'), message="Loaded")
    except Exception as e: return IndexStatusResponse(exists=False, document_count=0, collection_name="error", message=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not rag_chain: initialize_rag_system()
    if not rag_chain: return ChatResponse(response="", context_used=False, error="System not ready")
    try:
        # --- 1. LOG CONFIGURAÇÕES (Model, Temp, Sys_Prompt) ---
        llm_conf = APP_CONFIG.get('llm', {})
        logger.info("Chat Configuration", extra={
            "model_name": llm_conf.get('model_name'),
            "temperature": llm_conf.get('temperature'),
            "system_prompt": llm_conf.get('system_prompt')
        })

        # --- 2. LOG PROMPT ENVIADO ---
        logger.info(f"Prompt Enviado: {req.message}", extra={"prompt": req.message})
        
        res = run_chain_with_retry(rag_chain, req.message)
        
        # --- 3. LOG RESPOSTA/RETORNO ---
        logger.info(f"Resposta Gerada: {res}", extra={"response_content": res})
        
        return ChatResponse(response=res, context_used=True)
    except Exception as e:
         # Tratamento de erro melhorado: Desembrulha o RetryError
        real_error = e
        if isinstance(e, RetryError):
            real_error = e.last_attempt.exception()

        error_msg = str(e)
        logger.error(f"Chat Error: {error_msg}", exc_info=True) # Loga stack trace completo
        
        # Identifica 429 e avisa o frontend para esperar
        if "429" in error_msg or "TooManyRequests" in error_msg:
            return ChatResponse(
                response="", 
                context_used=False, 
                error="⚠️ Cota Excedida (429). O Google pediu para aguardar. Tente novamente em 60s.", 
                retry_after=60
            )
            
        return ChatResponse(response="", context_used=False, error="Erro interno do Chat.", retry_after=10)

@app.post("/api/ingest-pdf", response_model=IngestionResponse)
async def ingest_pdf(files: List[UploadFile] = File(...)):
    if not API_KEY: raise HTTPException(400, "No API Key")
    paths = []
    try:
        docs = []
        for file in files:
            if not file.filename.endswith('.pdf'): continue
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                shutil.copyfileobj(file.file, tmp)
                paths.append(tmp.name)
            docs.extend(PyPDFLoader(paths[-1]).load())
        if not docs: return IngestionResponse(status="error", message="No PDFs")
        
        c_size = APP_CONFIG.get('ingestion', {}).get('chunk_size', 1000)
        c_lap = APP_CONFIG.get('ingestion', {}).get('chunk_overlap', 200)
        chunks = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_lap).split_documents(docs)
        
        get_vectorstore().add_documents(chunks)
        initialize_rag_system()
        return IngestionResponse(status="success", message=f"Ingested {len(files)} files")
    except Exception as e:
        logger.error(f"Ingest Error: {e}")
        return IngestionResponse(status="error", message=str(e))
    finally:
        for p in paths: 
            if os.path.exists(p): os.remove(p)

@app.post("/api/ingest-url", response_model=IngestionResponse)
async def ingest_url(url: str = Form(...)):
    try:
        docs = WebBaseLoader(url).load()
        c_size = APP_CONFIG.get('ingestion', {}).get('chunk_size', 1000)
        c_lap = APP_CONFIG.get('ingestion', {}).get('chunk_overlap', 200)
        chunks = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_lap).split_documents(docs)
        get_vectorstore().add_documents(chunks)
        initialize_rag_system()
        return IngestionResponse(status="success", message="URL Ingested")
    except Exception as e: return IngestionResponse(status="error", message=str(e))

@app.get("/api/")
async def root():
    return {"message": "White Label API", "version": "6.0.0 (Unified Monolith)"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)