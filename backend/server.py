import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Optional

# Add parent directory to path to access root-level modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# Configuration
API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))

app = FastAPI(title="GeneXus AI Assistant API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for RAG chain
rag_chain = None
retriever = None

# Pydantic models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    context_used: bool
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    api_key_configured: bool
    database_loaded: bool
    message: str

class IndexStatusResponse(BaseModel):
    exists: bool
    document_count: int
    message: str

class IngestionRequest(BaseModel):
    source: str  # "pdf" or "web"

class IngestionResponse(BaseModel):
    status: str
    message: str
    progress: Optional[str] = None

# Initialize RAG system
def initialize_rag():
    """Initialize the RAG chain with error handling"""
    global rag_chain, retriever
    
    if not API_KEY:
        return {"success": False, "error": "GEMINI_API_KEY not configured"}
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=API_KEY
        )
        
        # Load vector store
        chroma_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), CHROMA_DB_PATH)
        vectorstore = Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings
        )
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.1,
            google_api_key=API_KEY
        )
        
        # Create prompt template
        PROMPT_TEMPLATE = """
You are the **GeneXus Code Assistant**, a senior GeneXus expert. Your mission is to provide complete and robust solutions, following best practices.

**CODE AND RESPONSE GUIDELINES:**
1.  **GeneXus Priority:** Always generate code **EXCLUSIVELY in GeneXus syntax**. Use code blocks (```genexus).
2.  **Focus on Structured Data:** Prioritize information found in **tables, property lists, and syntax definitions** within the 'CONTEXT'.
3.  **Contextual Inference:** If the 'CONTEXT' describes a process or data flow, **infer the logical flow** and translate it to the correct GeneXus syntax.
4.  **Strict Fidelity to Context (RAG):** Your response must be **entirely based on the provided 'CONTEXT'**.
5.  **Intelligent Rejection:** If the context is insufficient, decline to answer.
6.  **Language: Must interpret all languages but the response must always be in PT-BR or the language provided.

CONTEXT (GeneXus Documentation and Tutorials):
{context}

USER QUESTION: {question}
"""
        
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        
        def format_docs(docs):
            if not docs:
                return "No relevant context found."
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create RAG chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return {"success": True, "message": "RAG system initialized successfully"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    result = initialize_rag()
    if result["success"]:
        print("✅ RAG system initialized successfully")
    else:
        print(f"⚠️ RAG system initialization failed: {result.get('error')}")

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    api_configured = API_KEY is not None and API_KEY != ""
    db_loaded = rag_chain is not None
    
    status = "healthy" if (api_configured and db_loaded) else "degraded"
    
    message = "System operational"
    if not api_configured:
        message = "API key not configured. Check .env file."
    elif not db_loaded:
        message = "Vector database not loaded. Run ingestion scripts."
    
    return HealthResponse(
        status=status,
        api_key_configured=api_configured,
        database_loaded=db_loaded,
        message=message
    )

@app.get("/api/index-status", response_model=IndexStatusResponse)
async def index_status():
    """Check vector database index status"""
    try:
        chroma_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), CHROMA_DB_PATH)
        
        if not os.path.exists(chroma_path):
            return IndexStatusResponse(
                exists=False,
                document_count=0,
                message="Vector database not found. Run 'python ingest.py' or 'python ingest_site.py' first."
            )
        
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_community.vectorstores import Chroma
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=API_KEY
        )
        
        vectorstore = Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings
        )
        
        count = vectorstore._collection.count()
        
        return IndexStatusResponse(
            exists=True,
            document_count=count,
            message=f"Vector database loaded with {count} document chunks"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking index: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for RAG queries"""
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if not API_KEY:
        return ChatResponse(
            response="",
            context_used=False,
            error="API key not configured. Please set GEMINI_API_KEY in .env file."
        )
    
    if not rag_chain:
        return ChatResponse(
            response="",
            context_used=False,
            error="RAG system not initialized. Please check if vector database exists."
        )
    
    try:
        response = rag_chain.invoke(request.message)
        
        return ChatResponse(
            response=response,
            context_used=True,
            error=None
        )
        
    except Exception as e:
        return ChatResponse(
            response="",
            context_used=False,
            error=f"Error generating response: {str(e)}"
        )

@app.post("/api/ingest", response_model=IngestionResponse)
async def ingest_documents(request: IngestionRequest):
    """Trigger document ingestion"""
    import subprocess
    import threading
    
    if request.source not in ["pdf", "web"]:
        raise HTTPException(status_code=400, detail="Source must be 'pdf' or 'web'")
    
    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               f"ingest{'_site' if request.source == 'web' else ''}.py")
    
    if not os.path.exists(script_path):
        raise HTTPException(status_code=500, detail=f"Ingestion script not found: {script_path}")
    
    def run_ingestion():
        """Run ingestion in background"""
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                timeout=600  # 10 minute timeout
            )
            print(f"Ingestion completed: {result.returncode}")
            print(f"Output: {result.stdout}")
            if result.stderr:
                print(f"Errors: {result.stderr}")
            
            # Reinitialize RAG after ingestion
            initialize_rag()
        except Exception as e:
            print(f"Ingestion error: {e}")
    
    # Start ingestion in background thread
    thread = threading.Thread(target=run_ingestion, daemon=True)
    thread.start()
    
    source_name = "PDF documents" if request.source == "pdf" else "GeneXus website"
    return IngestionResponse(
        status="started",
        message=f"Ingestion from {source_name} started in background",
        progress="Processing documents... This may take several minutes."
    )

@app.get("/api/")
async def root():
    """Root endpoint"""
    return {
        "message": "GeneXus AI Assistant API",
        "version": "2.0.0",
        "endpoints": {
            "health": "/api/health",
            "chat": "/api/chat",
            "index_status": "/api/index-status",
            "ingest": "/api/ingest"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
