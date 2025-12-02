import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. ENVIRONMENT CONFIGURATION ---
# Load Gemini API key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

if not API_KEY:
    print("❌ ERROR: GEMINI_API_KEY not configured. Cannot load the DB.")
    print("💡 Copy .env.example to .env and add your Gemini API key.")
    exit(1)

# --- 2. INDEX INITIALIZATION AND LOADING ---
# Initialize embeddings (must be the same model used to create the index)
try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=API_KEY
    )
except Exception as e:
    print(f"❌ Error initializing embeddings: {e}")
    exit(1)

try:
    print(f"Loading ChromaDB from folder '{CHROMA_DB_PATH}'...")
    # 1. Load persisted Vector Store
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    
    # 2. Similarity search (simple query)
    # Search for a term relevant to the new content (GeneXus 18)
    query_term = "GeneXus 18 Super Apps"
    print(f"\n🔍 Searching for: '{query_term}'...")
    results = vectorstore.similarity_search(query_term, k=10)
    
    # Get collection count
    collection_count = vectorstore._collection.count()
    
    print(f"\n✅ Total documents (chunks) in index: {collection_count}")
    print("\n🔗 10 Examples of Indexed Articles (Checking Source URL):")
    print("-" * 60)
    
    # 3. Iterate over results to show the source
    if not results:
        print("⚠️ No results found for the query.")
        print("   This might mean the index is empty or doesn't contain relevant content.")
    else:
        for i, doc in enumerate(results, 1):
            source_url = doc.metadata.get('source', 'Source not found')
            
            # Classify the origin
            if "docs.genexus.com" in source_url:
                source_type = "🌐 WEB ARTICLE"
            elif source_url.endswith(".pdf"):
                source_type = "📄 LOCAL PDF"
            else:
                source_type = "❓ OTHER SOURCE"
            
            # Display origin and URL
            print(f"\n{i}. [{source_type}]")
            print(f"   Source: {source_url}")
            print(f"   Preview: \"{doc.page_content[:100]}...\"")
    
    print("-" * 60)
    print(f"\n✅ Index verification complete!")
    
except FileNotFoundError:
    print(f"\n❌ ChromaDB directory '{CHROMA_DB_PATH}' not found.")
    print("💡 Run 'python ingest.py' or 'python ingest_site.py' first to create the index.")
except Exception as e:
    print(f"\n❌ Failed to load ChromaDB: {e}")
    print("💡 Make sure you have run the ingestion script successfully.")
