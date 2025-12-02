import os
import sys
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Optional
import tempfile
import shutil

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

class IngestionResponse(BaseModel):
    status: str
    message: str
    progress: Optional[str] = None
    chunks_created: Optional[int] = None

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

@app.post("/api/ingest-pdf", response_model=IngestionResponse)
async def ingest_pdf_files(files: List[UploadFile] = File(...)):
    """Ingest uploaded PDF files"""
    if not API_KEY:
        raise HTTPException(status_code=400, detail="API key not configured")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_community.vectorstores import Chroma
        
        documents = []
        temp_files = []
        
        # Save uploaded files temporarily
        for file in files:
            if not file.filename.endswith('.pdf'):
                continue
                
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_files.append(temp_file.name)
            
            with open(temp_file.name, 'wb') as f:
                content = await file.read()
                f.write(content)
            
            # Load PDF
            loader = PyPDFLoader(temp_file.name)
            documents.extend(loader.load())
        
        if not documents:
            return IngestionResponse(
                status="error",
                message="No valid PDF documents found in uploaded files"
            )
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200"))
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=API_KEY
        )
        
        # Add to vector store
        chroma_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), CHROMA_DB_PATH)
        
        try:
            vectorstore = Chroma(
                persist_directory=chroma_path,
                embedding_function=embeddings
            )
            vectorstore.add_documents(chunks)
        except:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=chroma_path
            )
        
        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        
        # Reinitialize RAG
        initialize_rag()
        
        return IngestionResponse(
            status="success",
            message=f"Successfully ingested {len(files)} PDF file(s)",
            chunks_created=len(chunks)
        )
        
    except Exception as e:
        return IngestionResponse(
            status="error",
            message=f"Error during ingestion: {str(e)}"
        )

@app.post("/api/ingest-url", response_model=IngestionResponse)
async def ingest_from_url(url: str = Form(...)):
    """Ingest documentation from a specific URL"""
    if not API_KEY:
        raise HTTPException(status_code=400, detail="API key not configured")
    
    if not url or not url.startswith('http'):
        raise HTTPException(status_code=400, detail="Invalid URL provided")
    
    try:
        from langchain_community.document_loaders import WebBaseLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_community.vectorstores import Chroma
        
        # Load web content
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        if not documents:
            return IngestionResponse(
                status="error",
                message="No content could be extracted from the URL"
            )
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200"))
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=API_KEY
        )
        
        # Add to vector store
        chroma_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), CHROMA_DB_PATH)
        
        try:
            vectorstore = Chroma(
                persist_directory=chroma_path,
                embedding_function=embeddings
            )
            vectorstore.add_documents(chunks)
        except:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=chroma_path
            )
        
        # Reinitialize RAG
        initialize_rag()
        
        return IngestionResponse(
            status="success",
            message=f"Successfully ingested content from URL",
            chunks_created=len(chunks)
        )
        
    except Exception as e:
        return IngestionResponse(
            status="error",
            message=f"Error during ingestion: {str(e)}"
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
            "ingest_pdf": "/api/ingest-pdf",
            "ingest_url": "/api/ingest-url"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
