import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load API Key from .env file
load_dotenv()

# Configuration from environment variables with defaults
API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
DOCS_PATH = os.getenv("DOCS_PATH", "./docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

def run_ingestion():
    """Run the document ingestion process."""
    
    # Validate API Key
    if not API_KEY:
        print("❌ ERROR: GEMINI_API_KEY not configured. Check your .env file.")
        print("💡 Copy .env.example to .env and add your Gemini API key.")
        return
    
    # 1. Load Documents
    print(f"📂 Loading documents from '{DOCS_PATH}'...")
    
    # Create docs directory if it doesn't exist
    if not os.path.exists(DOCS_PATH):
        os.makedirs(DOCS_PATH)
        print(f"✅ Created directory: {DOCS_PATH}")
        print(f"💡 Please add PDF files to the '{DOCS_PATH}' folder and run again.")
        return
    
    documents = []
    
    # Iterate through all PDFs in the 'docs' folder
    pdf_files = [f for f in os.listdir(DOCS_PATH) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"⚠️ No PDF files found in '{DOCS_PATH}'. Aborting.")
        print(f"💡 Please add PDF files to the '{DOCS_PATH}' folder and run again.")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF file(s)")
    
    for filename in pdf_files:
        try:
            filepath = os.path.join(DOCS_PATH, filename)
            print(f"  📖 Loading: {filename}...")
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
        except Exception as e:
            print(f"  ❌ Error loading {filename}: {e}")
            continue
    
    if not documents:
        print("❌ No documents were successfully loaded. Aborting.")
        return
    
    print(f"✅ Loaded {len(documents)} page(s) total")
    
    # 2. Segmentation (Chunking)
    print(f"\n✂️ Splitting {len(documents)} pages into chunks...")
    print(f"   Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    
    try:
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")
    except Exception as e:
        print(f"❌ Error during text splitting: {e}")
        return
    
    # 3. Create Embeddings and Indexing
    print(f"\n🔄 Creating embeddings with GoogleGenerativeAI and indexing in ChromaDB...")
    
    try:
        # Robust model for creating text vectors
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=API_KEY
        )
        
        # Create Vector Store and store vectors locally
        print(f"   Saving to: {CHROMA_DB_PATH}")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        
        print(f"\n✅ Ingestion completed successfully!")
        print(f"📊 Vector database saved to: {CHROMA_DB_PATH}")
        print(f"📈 Total chunks indexed: {len(chunks)}")
        print(f"\n💡 You can now run 'streamlit run app.py' to use the assistant.")
        
    except Exception as e:
        print(f"❌ Error during embedding/indexing: {e}")
        return

if __name__ == "__main__":
    print("🚀 Starting GeneXus Documentation Ingestion\n")
    print("=" * 60)
    run_ingestion()
    print("=" * 60)
