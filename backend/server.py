import os
import sys
import logging
import json
from datetime import datetime
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

# --- CONFIGURAÇÃO DE AMBIENTE ---
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# --- FORMATADOR CUSTOMIZADO ---
class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        if not log_record.get('timestamp'):
            now = datetime.fromtimestamp(record.created) if hasattr(record, 'created') else datetime.now()
            log_record['timestamp'] = now.strftime('%Y-%m-%dT%H:%M:%S')
        if not log_record.get('level'):
            log_record['level'] = record.levelname.upper() if record.levelname else 'INFO'

# Definição do caminho do log
def resolve_log_file_path() -> str:
    env_path = os.getenv("LOG_FILE_PATH")
    if env_path: return env_path
    
    prod_path = "/var/log/supervisor/backend.out.log"
    if os.path.exists(prod_path): return prod_path
    
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend.out.log")
    if not os.path.exists(local_path):
        try:
            with open(local_path, 'a') as f: pass
        except: pass
    return local_path

CURRENT_LOG_FILE = resolve_log_file_path()

# --- FORMATADOR CUSTOMIZADO ---
class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        
        # Garante timestamp
        if not log_record.get('timestamp'):
            now = datetime.fromtimestamp(record.created) if hasattr(record, 'created') else datetime.now()
            log_record['timestamp'] = now.strftime('%Y-%m-%dT%H:%M:%S')
            
        # Garante level
        if not log_record.get('level'):
            log_record['level'] = record.levelname.upper() if record.levelname else 'INFO'

def setup_logging():
    """
    Configura logging para evitar duplicidade.
    Estratégia: Logger da aplicação escreve DIRETO no arquivo e NÃO propaga para o root/console.
    """
    # 1. Preparar Handlers
    # Console (apenas para erros críticos do sistema ou uvicorn startup)
    log_handler = logging.StreamHandler(sys.stdout)
    # Arquivo (onde o LogsViewer lê)
    file_handler = logging.FileHandler(CURRENT_LOG_FILE)
    
    formatter = CustomJsonFormatter(
        fmt='%(timestamp)s %(level)s %(name)s %(message)s %(module)s %(funcName)s %(lineno)d'
    )
    
    log_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 2. Configurar Logger da Aplicação (__name__)
    # IMPORTANTE: propagate=False impede que suba para o Root (evitando o console duplicado)
    app_logger = logging.getLogger(__name__)
    app_logger.setLevel(logging.INFO)
    app_logger.handlers = [file_handler] # Apenas arquivo!
    app_logger.propagate = False 
    
    # 3. Configurar Loggers do Uvicorn/FastAPI
    # Eles precisam ir para o arquivo também para aparecerem no viewer
    logging.getLogger('uvicorn.access').handlers = [file_handler]
    logging.getLogger('uvicorn.access').propagate = False
    
    logging.getLogger('uvicorn.error').handlers = [file_handler]
    logging.getLogger('uvicorn.error').propagate = False
    
    # Root Logger (Safety net para outras libs)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [log_handler] # Root continua no console para debug de crash
    
    return app_logger

logger = setup_logging()

# --- VARIÁVEIS GLOBAIS ---
API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS_PER_INGESTION", "50"))
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
DELAY_SECONDS = float(os.getenv("EMBEDDING_DELAY_SECONDS", "5"))
USE_CACHE = os.getenv("USE_EMBEDDINGS_CACHE", "true").lower() == "true"

rag_chain = None
vectorstore_instance = None 

# --- HELPERS ---

def get_optimized_embeddings():
    if not API_KEY: raise ValueError("GEMINI_API_KEY missing")
    base_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=API_KEY,
        transport="rest",
        task_type="retrieval_document"
    )
    return OptimizedEmbeddings(base_embeddings, use_cache=USE_CACHE, batch_size=BATCH_SIZE, delay_between_batches=DELAY_SECONDS)

def get_vectorstore():
    chroma_abs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), CHROMA_DB_PATH)
    return Chroma(persist_directory=chroma_abs_path, embedding_function=get_optimized_embeddings())

def initialize_rag_system():
    global rag_chain, vectorstore_instance
    if not API_KEY: return {"success": False, "error": "GEMINI_API_KEY missing"}

    try:
        vectorstore_instance = get_vectorstore()
        retriever = vectorstore_instance.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0.1,
            google_api_key=API_KEY,
            max_retries=1,              
            transport="rest"
        )
        
        PROMPT_TEMPLATE = """
You are the **GeneXus Code Assistant**, a senior GeneXus expert.
**LANGUAGE INSTRUCTION:** Answer in the same language as the user's question.

CONTEXT:
{context}

USER QUESTION: {question}
"""
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs) if docs else "No context."
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return {"success": True, "message": "RAG initialized (Gemini 2.0 Flash)"}
    except Exception as e:
        logger.error(f"RAG Init Error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=20), retry=retry_if_exception_type(Exception))
def run_chain_with_retry(chain, message):
    logger.info("Invoking chain...")
    return chain.invoke(message)

# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup initiated")
    result = initialize_rag_system()
    if result["success"]:
        logger.info("RAG ready", extra={"init_msg": result.get("message")})
    else:
        logger.error("RAG failed", extra={"init_err": result.get("error")})
    yield
    logger.info("Shutdown")

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

class IngestionResponse(BaseModel):
    status: str
    message: str
    chunks_created: Optional[int] = None

class HealthResponse(BaseModel):
    status: str
    api_key_configured: bool
    database_loaded: bool
    message: str

class IndexStatusResponse(BaseModel):
    exists: bool
    document_count: int
    message: str

# --- ENDPOINTS ---

@app.get("/api/logs")
async def get_logs(
    lines: int = 100, 
    level: Optional[str] = None, 
    search: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
):
    if not os.path.exists(CURRENT_LOG_FILE):
        return {"logs": [], "error": "Log file not found"}

    try:
        dt_start = None
        dt_end = None
        log_fmt = '%Y-%m-%dT%H:%M:%S'

        if start_time:
            try:
                clean_start = start_time.replace(' ', 'T').split('.')[0].replace('Z', '')
                dt_start = datetime.strptime(clean_start, log_fmt)
            except (ValueError, TypeError): pass

        if end_time:
            try:
                clean_end = end_time.replace(' ', 'T').split('.')[0].replace('Z', '')
                dt_end = datetime.strptime(clean_end, log_fmt)
            except (ValueError, TypeError): pass

        filtered_logs = []
        
        with open(CURRENT_LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    log_entry = json.loads(line)
                    
                    # 1. Filtro Level
                    log_level = log_entry.get('level') or ''
                    if level and log_level.upper() != level.upper(): continue
                    
                    # 2. Filtro Texto
                    if search and search.lower() not in str(log_entry).lower(): continue
                    
                    # 3. Filtro Data
                    if dt_start or dt_end:
                        ts_str = log_entry.get('timestamp')
                        if not ts_str: continue 
                        try:
                            log_dt = datetime.strptime(ts_str, log_fmt)
                            if dt_start and log_dt < dt_start: continue
                            if dt_end and log_dt > dt_end: continue
                        except (ValueError, TypeError): continue

                    filtered_logs.append(log_entry)
                except json.JSONDecodeError:
                    pass

        return {
            "logs": filtered_logs[-lines:], 
            "total_matches": len(filtered_logs),
            "source": CURRENT_LOG_FILE
        }

    except Exception as e:
        logger.error(f"Critical error in get_logs: {str(e)}", exc_info=True)
        return {"logs": [], "error": f"Server error reading logs: {str(e)}"}

@app.get("/api/ping-ai")
async def ping_ai():
    if not API_KEY: raise HTTPException(status_code=500, detail="API Key missing")
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=API_KEY, temperature=0, max_retries=1)
        res = llm.invoke("Pong")
        return {"status": "success", "reply": res.content}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    is_api_configured = bool(API_KEY and API_KEY.strip())
    is_db_loaded = rag_chain is not None
    status = "healthy" if (is_api_configured and is_db_loaded) else "degraded"
    msg = "System operational"
    if not is_api_configured: msg = "API Key missing"
    elif not is_db_loaded: msg = "RAG not initialized"

    return HealthResponse(
        status=status,
        api_key_configured=is_api_configured,
        database_loaded=is_db_loaded,
        message=msg
    )

@app.get("/api/index-status", response_model=IndexStatusResponse)
async def index_status():
    try:
        vs = vectorstore_instance or get_vectorstore()
        count = vs._collection.count()
        return IndexStatusResponse(exists=True, document_count=count, message=f"Loaded {count} docs")
    except Exception as e:
        return IndexStatusResponse(exists=False, document_count=0, message=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Endpoint principal de Chat com RAG e Logs de Prompt/Resposta"""
    
    # Validação básica
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Verifica inicialização
    if not rag_chain:
        init_result = initialize_rag_system()
        if not init_result["success"]:
            return ChatResponse(response="", context_used=False, error="RAG system not initialized")
    
    try:
        # 1. LOG DE ENTRADA (O Prompt do Usuário)
        # Usamos 'extra' para que o dado fique estruturado no JSON e visível no LogsViewer
        logger.info("Processing RAG Query", extra={
            "user_question": request.message,
            "user_ip": "client_ip_placeholder" # Em produção você pegaria do request.client.host
        })

        # Executa a cadeia (Chain)
        response = run_chain_with_retry(rag_chain, request.message)
        
        # 2. LOG DE SAÍDA (A Resposta da IA)
        logger.info("RAG Response generated", extra={
            "ai_response": response,
            "response_length": len(response)
        })

        return ChatResponse(response=response, context_used=True)
        
    except Exception as e:
        error_str = str(e)
        logger.error(f"Chat error after retries: {error_str}", extra={
            "failed_question": request.message
        })
        
        return ChatResponse(
            response="", 
            context_used=False, 
            error="O servidor está sobrecarregado (Rate Limit). Tente novamente em 1 minuto.",
            retry_after=60
        )

@app.post("/api/ingest-pdf", response_model=IngestionResponse)
async def ingest_pdf(files: List[UploadFile] = File(...)):
    logger.info(f"Ingesting {len(files)} PDFs")
    if not API_KEY: raise HTTPException(status_code=400, detail="API Key missing")
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
        
        chunks = RecursiveCharacterTextSplitter(chunk_size=int(os.getenv("CHUNK_SIZE", "1000")), chunk_overlap=200).split_documents(docs)
        get_vectorstore().add_documents(chunks)
        initialize_rag_system()
        return IngestionResponse(status="success", message=f"Ingested {len(files)} files", chunks_created=len(chunks))
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
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        get_vectorstore().add_documents(chunks)
        initialize_rag_system()
        return IngestionResponse(status="success", message="URL Ingested", chunks_created=len(chunks))
    except Exception as e:
        return IngestionResponse(status="error", message=str(e))

@app.get("/api/")
async def root():
    return {"message": "GeneXus AI API", "version": "3.3.1 (Date Fix)", "endpoints": ["chat", "logs"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)