import os
import json
import shutil
import tempfile
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse 
from tenacity import RetryError
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from app.config import APP_CONFIG, DEFAULT_CONFIG, API_KEY, GLOBAL_STATE, save_active_profile
from app.logging_config import logger, CURRENT_LOG_FILE
from app.schemas import ChatRequest, ChatResponse, IngestionResponse, HealthResponse, IndexStatusResponse, TextIngestRequest
from app.core import rag
from app.core.limiter import limiter

router = APIRouter()

# --- Classes Auxiliares ---
class ProfileSwitchRequest(BaseModel):
    profile_id: str

# --- Endpoints de Perfil e Config ---
@router.get("/config/profiles")
async def get_available_profiles():
    """Retorna lista de perfis disponíveis e qual está ativo."""
    return {
        "active": GLOBAL_STATE.get("active_profile"),
        "profiles": list(GLOBAL_STATE.get("profiles", {}).keys())
    }

@router.post("/config/switch")
async def switch_profile(req: ProfileSwitchRequest):
    """Troca o perfil ativo e reinicia o sistema RAG."""
    try:
        if req.profile_id not in GLOBAL_STATE.get("profiles", {}):
            raise HTTPException(status_code=400, detail="Profile not found")
        
        # 1. Salva e Atualiza APP_CONFIG
        save_active_profile(req.profile_id)
        logger.info(f"Switched to profile: {req.profile_id}")
        
        # 2. Reinicia o RAG para pegar a nova collection_name e system_prompt
        # Forçamos a reinicialização zerando as variáveis globais do módulo rag
        rag.rag_chain = None
        rag.vectorstore_instance = None
        result = rag.initialize_rag_system()
        
        return {"status": "success", "message": f"Switched to {req.profile_id}", "rag_status": result}
    except Exception as e:
        logger.error(f"Error switching profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Endpoint de Logs ---
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
        with open(CURRENT_LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    entry = json.loads(line)
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
        coll_name = APP_CONFIG.get('storage',{}).get('collection_name')
        
        # Lógica Simples para ChromaDB
        if hasattr(vs, '_collection'):
            count = vs._collection.count()
        else:
            count = 0
            
        return IndexStatusResponse(
            exists=True, 
            document_count=count, 
            collection_name=coll_name, 
            message="Loaded (ChromaDB)"
        )
    except Exception as e: 
        return IndexStatusResponse(exists=False, document_count=0, collection_name="error", message=str(e))

@router.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    try:
        # Usa a função do rag.py para pegar o objeto de histórico correto
        history_obj = rag.get_session_history(session_id)
        messages = history_obj.messages
        
        # Formata para o Frontend
        formatted_history = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            formatted_history.append({
                "role": role,
                "content": msg.content,
                "timestamp": datetime.now().isoformat() # Data aproximada
            })
        return {"history": formatted_history}
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return {"history": []}

# --- CHAT STREAMING ---
@router.post("/chat/stream")
@limiter.limit("10/minute")
async def chat_stream(request: Request, req: ChatRequest):
    if not rag.rag_chain: rag.initialize_rag_system()
    if not rag.rag_chain: return ChatResponse(response="", context_used=False, error="System not ready")
    
    session_id = req.session_id or "default_session"
    active_model = rag.get_current_model_name()
    
    logger.info(f"Stream Request [{session_id}]", extra={
        "prompt": req.message, 
        "active_model": active_model, 
        "fallback_enabled": True
    })

    async def event_generator():
        try:
            # Obtém o gerador (pode disparar troca de modelo se falhar na conexão inicial)
            stream_iterator = await rag.run_chain_with_fallback(
                {"question": req.message},
                config={"configurable": {"session_id": session_id}},
                is_streaming=True
            )

            async for chunk in stream_iterator:
                if "response" in chunk and chunk["response"]:
                    data = json.dumps({"type": "token", "content": chunk["response"]})
                    yield data + "\n"
                
                if "sources" in chunk and chunk["sources"]:
                    unique_sources = set()
                    for doc in chunk["sources"]:
                        src = doc.metadata.get("source", "Desconhecido")
                        unique_sources.add(os.path.basename(src))
                    
                    if unique_sources:
                        data = json.dumps({"type": "sources", "content": list(unique_sources)})
                        yield data + "\n"

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Stream Error Final: {error_msg}")
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                user_msg = "⚠️ Todos os modelos de IA estão ocupados/sem cota no momento. Tente mais tarde."
            else:
                user_msg = f"Erro no sistema: {error_msg[:100]}"
            yield json.dumps({"type": "error", "content": user_msg}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

# --- CHAT STANDARD (LEGADO) ---
@router.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(request: Request, req: ChatRequest):
    if not rag.rag_chain: rag.initialize_rag_system()
    if not rag.rag_chain: return ChatResponse(response="", context_used=False, error="System not ready")
    try:
        session_id = req.session_id or "default_session"
        active_model = rag.get_current_model_name()
        
        logger.info("Chat Request", extra={
            "prompt": req.message,
            "session_id": session_id,
            "active_model": active_model
        })
        
        result = await rag.run_chain_with_fallback(
            {"question": req.message}, 
            config={"configurable": {"session_id": session_id}},
            is_streaming=False
        )
        
        answer_text = result.get("response", "")
        source_docs = result.get("sources", [])
        unique_sources = set()
        for doc in source_docs:
            src = doc.metadata.get("source", "Desconhecido")
            filename = os.path.basename(src)
            unique_sources.add(filename)
        
        if unique_sources:
            footer = "\n\n---\n📚 **Fontes Consultadas:**\n" + "\n".join([f"- `{s}`" for s in unique_sources])
            answer_text += footer

        return ChatResponse(response=answer_text, context_used=True)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Chat Standard Error: {error_msg}")
        return ChatResponse(response="", context_used=False, error="Erro ao processar (ver logs)")

@router.post("/ingest-text", response_model=IngestionResponse)
@limiter.limit("5/minute")
async def ingest_text(request: Request, body: TextIngestRequest):
    """Recebe texto bruto, fragmenta e salva no banco vetorial."""
    try:
        if not body.text or len(body.text.strip()) < 10:
             return IngestionResponse(status="error", message="Texto muito curto ou vazio.")

        # Cria um Documento LangChain manual
        doc = Document(
            page_content=body.text,
            metadata={
                "source": f"[TEXT] {body.title}",
                "title": body.title,
                "created_at": datetime.now().isoformat()
            }
        )
        
        # Configuração de Chunking
        c_size = APP_CONFIG.get('ingestion', {}).get('chunk_size', 1000)
        c_lap = APP_CONFIG.get('ingestion', {}).get('chunk_overlap', 200)
        
        # Splitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_lap)
        chunks = splitter.split_documents([doc])
        
        # Salva no Banco
        rag.get_vectorstore().add_documents(chunks)
        
        # Reinicia sistema para pegar novos dados
        rag.initialize_rag_system()
        
        return IngestionResponse(status="success", message=f"Text Ingested: {body.title} ({len(chunks)} chunks)")
        
    except Exception as e:
        logger.error(f"Text Ingest Error: {e}")
        return IngestionResponse(status="error", message=f"Erro ao processar texto: {str(e)}")

@router.post("/ingest-pdf", response_model=IngestionResponse)
@limiter.limit("5/minute")
async def ingest_pdf(request: Request, files: List[UploadFile] = File(...)):
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
@limiter.limit("5/minute")
async def ingest_url(request: Request, url: str = Form(...)):
    try:
        loader = WebBaseLoader(web_path=url, header_template={"User-Agent": "Mozilla/5.0"})
        docs = loader.load()
        if not docs: return IngestionResponse(status="error", message="Conteúdo vazio.")
        c_size = APP_CONFIG.get('ingestion', {}).get('chunk_size', 1000)
        c_lap = APP_CONFIG.get('ingestion', {}).get('chunk_overlap', 200)
        chunks = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_lap).split_documents(docs)
        rag.get_vectorstore().add_documents(chunks)
        rag.initialize_rag_system()
        return IngestionResponse(status="success", message=f"URL Ingested")
    except Exception as e: return IngestionResponse(status="error", message=f"Erro: {str(e)}")

# --- INGESTÃO YOUTUBE ROBUSTA (Com Fallback Pytube -> Transcript) ---
@router.post("/ingest-youtube", response_model=IngestionResponse)
@limiter.limit("5/minute")
async def ingest_youtube(request: Request, url: str = Form(...)):
    try:
        docs = []
        video_title = "Video YouTube"
        
        # Tentativa 1: Com Metadados (Pytube - Instável)
        try:
            loader = YoutubeLoader.from_youtube_url(
                url, 
                add_video_info=True,
                language=["pt", "en", "es"],
                translation="pt"
            )
            docs = loader.load()
            if docs:
                video_title = docs[0].metadata.get('title', video_title)
        except Exception as e:
            # Se der erro (ex: 400 Bad Request do Youtube), loga e tenta sem metadados
            logger.warning(f"Falha ao obter metadados (Pytube): {e}. Tentando modo somente-texto...")
            
            # Tentativa 2: Sem Metadados (Apenas Transcrição - Estável)
            loader = YoutubeLoader.from_youtube_url(
                url, 
                add_video_info=False, # <--- Desliga o Pytube
                language=["pt", "en", "es"],
                translation="pt"
            )
            docs = loader.load()
        
        if not docs: 
            return IngestionResponse(status="error", message="Não foi possível extrair a transcrição (Legendas não encontradas).")

        # Adiciona tags para identificação no Admin
        for doc in docs:
            # Se não pegou título na tentativa 1, usa genérico
            t = doc.metadata.get("title", video_title)
            doc.metadata["source"] = f"[YOUTUBE] {t}"
            doc.metadata["title"] = t

        c_size = APP_CONFIG.get('ingestion', {}).get('chunk_size', 1000)
        c_lap = APP_CONFIG.get('ingestion', {}).get('chunk_overlap', 200)
        chunks = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_lap).split_documents(docs)
        
        rag.get_vectorstore().add_documents(chunks)
        rag.initialize_rag_system()
        
        return IngestionResponse(status="success", message=f"Video Ingested: {video_title}")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"YouTube Ingest Critical Error: {error_msg}")
        
        # Tratamento específico para bloqueio de IP (comum em Cloud/AWS/GCP)
        if "blocking requests" in error_msg:
             return IngestionResponse(
                 status="error", 
                 message="YouTube bloqueou o IP deste servidor (Cloud Block). Tente rodar localmente."
             )
             
        return IngestionResponse(status="error", message=f"Erro crítico ao processar vídeo: {str(e)}")