import os
import json
import shutil
import tempfile
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from tenacity import RetryError

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import APP_CONFIG, DEFAULT_CONFIG, API_KEY
from app.logging_config import logger, CURRENT_LOG_FILE # Importamos o caminho do log
from app.schemas import ChatRequest, ChatResponse, IngestionResponse, HealthResponse, IndexStatusResponse
from app.core import rag

router = APIRouter()

# --- Endpoint de Logs Recuperado ---
@router.get("/logs")
async def get_logs(lines: int = 100, level: Optional[str] = None, search: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None):
    if not os.path.exists(CURRENT_LOG_FILE): 
        return {"logs": [], "error": "Log file missing", "source": CURRENT_LOG_FILE}
    
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
        # Lê o arquivo de trás para frente seria ideal, mas aqui lemos tudo e filtramos
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
        
        # Retorna as últimas 'lines' linhas
        return {"logs": filtered[-lines:], "total_matches": len(filtered), "source": CURRENT_LOG_FILE}
    except Exception as e:
        logger.error(f"Log Error: {e}")
        return {"logs": [], "error": str(e)}

@router.get("/config")
async def get_frontend_config():
    """Retorna configurações visuais para o Frontend"""
    return APP_CONFIG.get('identity', DEFAULT_CONFIG['identity'])

@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if rag.rag_chain else "degraded",
        api_key_configured=bool(API_KEY),
        database_loaded=rag.rag_chain is not None,
        message="OK",
        app_name=APP_CONFIG.get('identity', {}).get('app_name', 'Unknown')
    )

@router.get("/index-status", response_model=IndexStatusResponse)
async def index_status():
    try:
        vs = rag.vectorstore_instance or rag.get_vectorstore()
        # ChromaDB check
        if hasattr(vs, '_collection'):
            count = vs._collection.count()
        else:
            count = 0
            
        coll_name = APP_CONFIG.get('storage',{}).get('collection_name')
        return IndexStatusResponse(exists=True, document_count=count, collection_name=coll_name, message="Loaded")
    except Exception as e: 
        return IndexStatusResponse(exists=False, document_count=0, collection_name="error", message=str(e))

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not rag.rag_chain: 
        rag.initialize_rag_system()
    
    if not rag.rag_chain: 
        return ChatResponse(response="", context_used=False, error="System not ready")
    
    try:
        # 1. Log Configurações
        llm_conf = APP_CONFIG.get('llm', {})
        logger.info("Chat Configuration", extra={
            "model_name": llm_conf.get('model_name'),
            "temperature": llm_conf.get('temperature'),
            "system_prompt": llm_conf.get('system_prompt')
        })

        # 2. Log Prompt
        logger.info(f"Prompt Enviado: {req.message}", extra={"prompt": req.message})
        
        res = rag.run_chain_with_retry(rag.rag_chain, req.message)
        
        # 3. Log Resposta
        logger.info(f"Resposta Gerada: {res}", extra={"response_content": res})
        
        return ChatResponse(response=res, context_used=True)
    except Exception as e:
        real_error = e
        if isinstance(e, RetryError):
            real_error = e.last_attempt.exception()

        error_msg = str(e)
        logger.error(f"Chat Error: {error_msg}", exc_info=True)
        
        if "429" in error_msg or "TooManyRequests" in error_msg:
            return ChatResponse(
                response="", 
                context_used=False, 
                error="⚠️ Cota Excedida (429). Aguarde 60s.", 
                retry_after=60
            )
            
        return ChatResponse(response="", context_used=False, error="Erro interno do Chat.", retry_after=10)

@router.post("/ingest-pdf", response_model=IngestionResponse)
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
        
        rag.get_vectorstore().add_documents(chunks)
        rag.initialize_rag_system()
        
        return IngestionResponse(status="success", message=f"Ingested {len(files)} files")
    except Exception as e:
        logger.error(f"Ingest Error: {e}")
        return IngestionResponse(status="error", message=str(e))
    finally:
        for p in paths: 
            if os.path.exists(p): os.remove(p)

@router.post("/ingest-url", response_model=IngestionResponse)
async def ingest_url(url: str = Form(...)):
    try:
        docs = WebBaseLoader(url).load()
        c_size = APP_CONFIG.get('ingestion', {}).get('chunk_size', 1000)
        c_lap = APP_CONFIG.get('ingestion', {}).get('chunk_overlap', 200)
        chunks = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_lap).split_documents(docs)
        rag.get_vectorstore().add_documents(chunks)
        rag.initialize_rag_system()
        return IngestionResponse(status="success", message="URL Ingested")
    except Exception as e: return IngestionResponse(status="error", message=str(e))