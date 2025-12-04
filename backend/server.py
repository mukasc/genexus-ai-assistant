import os
import sys
import logging
import shutil
import tempfile
import json
import time
from contextlib import asynccontextmanager
from typing import List, Optional

# Third-party imports
from pythonjsonlogger import jsonlogger
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Add parent directory to path to access root-level modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import rate limiting optimizer
from gemini_optimizer import OptimizedEmbeddings, get_embedding_stats

# --- CONFIGURAÇÃO DE AMBIENTE E LOGS ---
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "backend.out.log")

def setup_logging():
    """Configure structured JSON logging for the application"""
    log_handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        fmt='%(timestamp)s %(level)s %(name)s %(message)s %(module)s %(funcName)s %(lineno)d',
        rename_fields={'levelname': 'level', 'asctime': 'timestamp'},
        datefmt='%Y-%m-%dT%H:%M:%S'
    )
    log_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [log_handler]
    
    # Configure uvicorn loggers
    for logger_name in ['uvicorn', 'uvicorn.access', 'uvicorn.error']:
        l = logging.getLogger(logger_name)
        l.handlers = [log_handler]
        l.propagate = False
    
    return logging.getLogger(__name__)

logger = setup_logging()

# --- VARIÁVEIS GLOBAIS E CONFIG ---
API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))

# Rate limiting configuration
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS_PER_INGESTION", "50"))
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
DELAY_SECONDS = float(os.getenv("EMBEDDING_DELAY_SECONDS", "5"))
USE_CACHE = os.getenv("USE_EMBEDDINGS_CACHE", "true").lower() == "true"

# Estado Global
rag_chain = None
vectorstore_instance = None 

# --- HELPERS (CLEAN CODE) ---

def resolve_log_file_path() -> Optional[str]:
    """Tenta encontrar o arquivo de log em múltiplos locais."""
    env_path = os.getenv("LOG_FILE_PATH")
    if env_path and os.path.exists(env_path): return env_path
    
    prod_path = "/var/log/supervisor/backend.out.log"
    if os.path.exists(prod_path): return prod_path
    
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend.out.log")
    if os.path.exists(local_path): return local_path
        
    return None

def get_optimized_embeddings():
    """Fábrica que retorna os Embeddings com Cache e Rate Limit ativados"""
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY not configured")
        
    # Mantemos o modelo de embedding padrão (geralmente funciona com todos)
    base_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=API_KEY,
        transport="rest",
        task_type="retrieval_document"
    )
    
    return OptimizedEmbeddings(
        base_embeddings,
        use_cache=USE_CACHE,
        batch_size=BATCH_SIZE,
        delay_between_batches=DELAY_SECONDS
    )

def get_vectorstore():
    """Retorna a instância do ChromaDB configurada com Embeddings Otimizados"""
    chroma_abs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), CHROMA_DB_PATH)
    embeddings = get_optimized_embeddings()
    
    return Chroma(
        persist_directory=chroma_abs_path,
        embedding_function=embeddings
    )

def initialize_rag_system():
    """Inicializa o sistema RAG"""
    global rag_chain, vectorstore_instance
    
    if not API_KEY:
        return {"success": False, "error": "GEMINI_API_KEY not configured"}

    try:
        # 1. Setup VectorStore
        vectorstore_instance = get_vectorstore()
        retriever = vectorstore_instance.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        
        # 2. Setup LLM
        # ATUALIZAÇÃO: Usando modelo confirmado na sua lista
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0.1,
            google_api_key=API_KEY,
            max_retries=1,              
            transport="rest"
        )
        
        # 3. Setup Prompt
        PROMPT_TEMPLATE = """
You are the **GeneXus Code Assistant**, a senior GeneXus expert. Your mission is to provide complete and robust solutions.

CONTEXT (GeneXus Documentation):
{context}

USER QUESTION: {question}
"""
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs) if docs else "No relevant context found."
        
        # 4. Create Chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return {"success": True, "message": "RAG system initialized (Model: gemini-2.0-flash)"}
        
    except Exception as e:
        logger.error(f"RAG Init Error: {e}")
        return {"success": False, "error": str(e)}

# --- FUNÇÃO DE EXECUÇÃO SEGURA ---
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=20),
    retry=retry_if_exception_type(Exception)
)
def run_chain_with_retry(chain, message):
    logger.info("Attempting to invoke chain...")
    return chain.invoke(message)

# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup initiated")
    result = initialize_rag_system()
    if result["success"]:
        logger.info("RAG system initialized", extra={"init_message": result.get("message")})
    else:
        logger.error("RAG init failed", extra={"init_error": result.get("error")})
    yield
    logger.info("Application shutdown")

# --- APP CONFIG ---
app = FastAPI(title="GeneXus AI Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELS ---
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    context_used: bool
    error: Optional[str] = None
    retry_after: Optional[int] = None

class HealthResponse(BaseModel):
    status: str
    api_key_configured: bool
    database_loaded: bool
    message: str

class IndexStatusResponse(BaseModel):
    exists: bool
    document_count: int
    message: str

class IngestionResponse(BaseModel):
    status: str
    message: str
    chunks_created: Optional[int] = None

# --- ENDPOINTS ---

@app.get("/api/ping-ai")
async def ping_ai():
    """Testa apenas a conexão com o Gemini (sem RAG/VectorStore)"""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API Key not configured")
    
    try:
        # ATUALIZAÇÃO: Usando modelo confirmado
        test_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=API_KEY,
            temperature=0,
            max_retries=1
        )
        response = test_llm.invoke("Responda apenas com a palavra 'Pong'")
        return {
            "status": "success", 
            "message": "Connection established", 
            "ai_reply": response.content,
            "model_used": "gemini-2.0-flash"
        }
    except Exception as e:
        logger.error(f"Ping AI failed: {e}")
        return {
            "status": "error", 
            "message": "Could not connect to Gemini API", 
            "detail": str(e)
        }

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    api_configured = bool(API_KEY)
    db_loaded = rag_chain is not None
    status = "healthy" if (api_configured and db_loaded) else "degraded"
    
    return HealthResponse(
        status=status,
        api_key_configured=api_configured,
        database_loaded=db_loaded,
        message="System operational" if status == "healthy" else "Check logs/config"
    )

@app.get("/api/index-status", response_model=IndexStatusResponse)
async def index_status():
    try:
        vs = get_vectorstore()
        count = vs._collection.count()
        return IndexStatusResponse(
            exists=True, 
            document_count=count, 
            message=f"Vector database loaded with {count} chunks"
        )
    except Exception as e:
        return IndexStatusResponse(exists=False, document_count=0, message=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if not rag_chain:
        init_result = initialize_rag_system()
        if not init_result["success"]:
            return ChatResponse(response="", context_used=False, error="RAG system not initialized")
    
    try:
        response = run_chain_with_retry(rag_chain, request.message)
        return ChatResponse(response=response, context_used=True)
        
    except Exception as e:
        error_str = str(e)
        logger.error(f"Chat error after retries: {error_str}")
        return ChatResponse(
            response="", 
            context_used=False, 
            error="O servidor está sobrecarregado (Rate Limit). Tente novamente em 1 minuto.",
            retry_after=60
        )

@app.post("/api/ingest-pdf", response_model=IngestionResponse)
async def ingest_pdf_files(files: List[UploadFile] = File(...)):
    if not API_KEY:
        raise HTTPException(status_code=400, detail="API key not configured")
    
    temp_files = []
    try:
        documents = []
        for file in files:
            if not file.filename.endswith('.pdf'): continue
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_files.append(temp_file.name)
            with open(temp_file.name, 'wb') as f:
                content = await file.read()
                f.write(content)
            
            loader = PyPDFLoader(temp_file.name)
            documents.extend(loader.load())

        if not documents:
            return IngestionResponse(status="error", message="No valid PDF documents found")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200"))
        )
        chunks = text_splitter.split_documents(documents)

        vs = get_vectorstore()
        vs.add_documents(chunks) 
        
        initialize_rag_system()
        
        return IngestionResponse(status="success", message=f"Ingested {len(files)} files", chunks_created=len(chunks))

    except Exception as e:
        logger.error(f"PDF ingestion error: {e}")
        return IngestionResponse(status="error", message=str(e))
    finally:
        for tf in temp_files:
            if os.path.exists(tf): os.remove(tf)

@app.post("/api/ingest-url", response_model=IngestionResponse)
async def ingest_from_url(url: str = Form(...)):
    if not API_KEY:
        raise HTTPException(status_code=400, detail="API key not configured")

    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        if not documents:
            return IngestionResponse(status="error", message="No content extracted")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200"))
        )
        chunks = text_splitter.split_documents(documents)
        
        vs = get_vectorstore()
        vs.add_documents(chunks)
        
        initialize_rag_system()
        
        return IngestionResponse(status="success", message="URL Ingested", chunks_created=len(chunks))

    except Exception as e:
        logger.error(f"URL ingestion error: {e}")
        return IngestionResponse(status="error", message=str(e))

@app.get("/api/logs")
async def get_logs(lines: int = 100, level: Optional[str] = None, search: Optional[str] = None):
    log_file_path = resolve_log_file_path()
    
    if not log_file_path:
        return {
            "logs": [], 
            "error": "Log file not found. Checked: ENV, /var/log/supervisor/, and local dir.",
            "path_searched": "/var/log/supervisor/backend.out.log"
        }

    try:
        logs = []
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            file_lines = f.readlines()
            
        last_lines = file_lines[-lines:] if lines > 0 else file_lines
        
        for line in last_lines:
            if not line.strip(): continue
            try:
                log_entry = json.loads(line)
                if level and log_entry.get('level') != level: continue
                if search and search.lower() not in str(log_entry).lower(): continue
                logs.append(log_entry)
            except json.JSONDecodeError:
                if search and search.lower() not in line.lower(): continue
                logs.append({"message": line.strip(), "level": "system", "timestamp": "unknown"})
                
        return {"logs": logs, "total": len(logs), "source": log_file_path}

    except Exception as e:
        logger.error(f"Error reading logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error reading log: {str(e)}")

@app.get("/api/")
async def root():
    return {
        "message": "GeneXus AI Assistant API",
        "version": "2.7.0 (Stable Gemini 2.0)",
        "endpoints": {
            "health": "/api/health",
            "chat": "/api/chat",
            "logs": "/api/logs",
            "ping": "/api/ping-ai"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)