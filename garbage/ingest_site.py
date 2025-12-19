import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
import re
import requests
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- 1. ENVIRONMENT SETUP AND API KEY ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Chrome driver path from environment (optional)
CHROME_DRIVER_PATH = os.getenv("CHROME_DRIVER_PATH", "")

# Configuration
MAX_ARTICLES_TO_INDEX = int(os.getenv("MAX_ARTICLES_TO_INDEX", "50"))
MAX_PAGES_TO_SCAN = int(os.getenv("MAX_PAGES_TO_SCAN", "10"))

if not API_KEY:
    print("❌ ERROR: GEMINI_API_KEY environment variable is not configured.")
    print("💡 Copy .env.example to .env and add your Gemini API key.")
    exit(1)

# URLs (PAGINATED SEARCH STRATEGY)
URL_SEARCH_BASE = "https://docs.genexus.com/en/hsearch?+category%3AGeneXus+18+Help"
BASE_DOCS_URL = "https://docs.genexus.com"


def get_driver_with_selenium():
    """Initialize and configure Chrome driver."""
    
    options = Options()
    
    # Configuration to emulate a real browser
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--disable-gpu")
    
    try:
        if CHROME_DRIVER_PATH and os.path.exists(CHROME_DRIVER_PATH):
            service = Service(CHROME_DRIVER_PATH)
            driver = webdriver.Chrome(service=service, options=options)
        else:
            # Try to use system PATH
            driver = webdriver.Chrome(options=options)
        
        # Implicit wait for elements to appear (useful for JS)
        driver.implicitly_wait(5)
        return driver
    except Exception as e:
        print(f"❌ Error initializing Chrome driver: {e}")
        print("💡 Make sure Chrome/Chromium and chromedriver are installed.")
        print("💡 Download chromedriver from: https://chromedriver.chromium.org/")
        raise


def run_ingestion():
    """Run the web documentation ingestion process."""
    
    # List to store new Web documents
    documents = []
    article_links = set()
    driver = None

    # --- 2. WEB DOCUMENTATION LOADING (SELENIUM and Pagination) ---
    print(f"\n🕷️ Starting Scraper on GeneXus 18 Search pages (up to {MAX_PAGES_TO_SCAN} pages)...")
    print(f"   Target: {MAX_ARTICLES_TO_INDEX} articles\n")
    
    current_page = 1
    
    try:
        # Initialize driver outside loop to reuse it
        print("🚀 Initializing browser...")
        driver = get_driver_with_selenium()
        print("✅ Browser ready\n")

        while current_page <= MAX_PAGES_TO_SCAN and len(article_links) < MAX_ARTICLES_TO_INDEX:
            
            # 1. Build Paginated URL
            if current_page == 1:
                page_url = URL_SEARCH_BASE
            else:
                page_url = f"{URL_SEARCH_BASE},{current_page}"
            
            print(f"📄 Processing Page {current_page}: {page_url}")

            # 2. Navigate and wait for content to load
            try:
                driver.get(page_url)
                
                # Wait for search results to load (more efficient than sleep)
                wait = WebDriverWait(driver, 15)
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'span.Search__Title > a')))
                
                # Small additional wait for dynamic content
                time.sleep(2)  # Reduced from 10 seconds
                
            except Exception as e:
                print(f"  ⚠️ Timeout waiting for page to load: {e}")
                break
            
            # 3. Direct Link Extraction
            CSS_SELECTOR = 'span.Search__Title > a'
            
            try:
                links_elements = driver.find_elements(By.CSS_SELECTOR, CSS_SELECTOR)
            except Exception as e:
                print(f"  ❌ Error finding links: {e}")
                break
            
            if not links_elements:
                if current_page > 1:
                    print("  ℹ️ No more links found. End of pagination.")
                else:
                    print("  ⚠️ No links found on first page.")
                break

            print(f"  🔗 {len(links_elements)} article links found on page {current_page}")

            new_links_on_page = 0
            for link_element in links_elements:
                try:
                    relative_url = link_element.get_attribute('href')
                    
                    # Filter to ensure it's a wiki article and not a file or anchor
                    if (relative_url and 
                        '/en/wiki?' in relative_url and 
                        '#' not in relative_url and 
                        not any(ext in relative_url for ext in ['.png', '.jpg', '.gif', '.css', '.js', '.svg'])
                    ):
                        full_url = relative_url
                        
                        # Ensure URL is absolute
                        if full_url.startswith('/'):
                            full_url = BASE_DOCS_URL + full_url

                        # Add link if not duplicate
                        if full_url not in article_links:
                            article_links.add(full_url)
                            new_links_on_page += 1
                            
                            if len(article_links) >= MAX_ARTICLES_TO_INDEX:
                                break
                except Exception as e:
                    print(f"  ⚠️ Error processing link: {e}")
                    continue
            
            print(f"  ➕ {new_links_on_page} new links added. Total: {len(article_links)}/{MAX_ARTICLES_TO_INDEX}\n")

            if len(article_links) >= MAX_ARTICLES_TO_INDEX:
                print("✅ Reached maximum article limit")
                break
            
            # Prepare for next iteration
            current_page += 1

    except Exception as e:
        print(f"❌ ERROR during navigation or extraction: {e}. Interrupting.")
    finally:
        # 4. Close driver
        if driver:
            driver.quit()
            print("🔒 Browser closed\n")
    
    # --- Continuation of Ingestion ---
    
    if not article_links:
        print("\n⚠️ No article links were extracted. Ending ingestion.")
        return

    print(f"✅ Successfully extracted {len(article_links)} unique article URLs\n")
    
    # 3. Load COMPLETE CONTENT of each selected article
    article_documents = []
    print("📥 Starting to load content from selected articles...\n")
    
    for i, link in enumerate(article_links, 1):
        print(f"  {i}/{len(article_links)}: Indexing {link}")
        
        # WebBaseLoader is used to load article content (URL by URL)
        try:
            article_loader = WebBaseLoader(link)
            article_documents.extend(article_loader.load())
        except Exception as e:
            print(f"    ❌ ERROR loading {link}: {e}")
    
    documents.extend(article_documents)
    print(f"\n✅ Total complete articles loaded: {len(article_documents)}\n")

    if not documents:
        print("⚠️ No Web documents were loaded. Ending ingestion.")
        return

    # --- 3. CHUNKING ---
    print(f"✂️ Splitting {len(documents)} new documents into chunks...")
    print(f"   Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    try:
        new_chunks = text_splitter.split_documents(documents)
        print(f"✅ Created {len(new_chunks)} new chunks\n")
    except Exception as e:
        print(f"❌ Error during text splitting: {e}")
        return

    # --- 4. EMBEDDING AND STORAGE (CHROMA DB) ---
    print("🔄 Initializing Embeddings and handling Chroma DB...")
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=API_KEY
        )
    except Exception as e:
        print(f"❌ Error initializing embeddings: {e}")
        return

    try:
        # Try to load existing database (to combine)
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
        print("✅ Existing database (including PDFs) loaded successfully.")
        
        # Add new Web chunks to existing database
        print(f"➕ Adding {len(new_chunks)} new Web documents to existing database...")
        vectorstore.add_documents(new_chunks)
        
    except Exception as e:
        # If database doesn't exist, create a new one
        print(f"ℹ️ No existing database found. Creating new database from scratch...")
        vectorstore = Chroma.from_documents(
            documents=new_chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
    
    print(f"\n✅ Ingestion completed successfully!")
    print(f"📊 Vector database saved to: {CHROMA_DB_PATH}")
    print(f"📈 Total new chunks indexed: {len(new_chunks)}")
    print(f"\n💡 You can now run 'streamlit run app.py' to use the assistant.")
    print(f"💡 Run 'python check_index.py' to verify the indexed content.")

if __name__ == "__main__":
    print("🚀 Starting GeneXus Web Documentation Ingestion\n")
    print("=" * 70)
    run_ingestion()
    print("=" * 70)
