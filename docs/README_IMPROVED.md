# 🤖 GeneXus AI Assistant (RAG)

An intelligent chatbot assistant specialized in GeneXus development, powered by RAG (Retrieval Augmented Generation) architecture.

## 🌟 Features

- **Specialized Knowledge**: Trained on GeneXus documentation
- **RAG Architecture**: Combines vector search with generative AI
- **Multiple Data Sources**: Supports PDF documents and web scraping
- **Image Processing**: Can extract and describe images from PDFs using Gemini Vision
- **Interactive Chat**: Streamlit-based user interface
- **Persistent Storage**: ChromaDB for vector embeddings

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini 2.0 Flash
- **Vector Database**: ChromaDB
- **Framework**: LangChain
- **Web Scraping**: Selenium + BeautifulSoup
- **Document Processing**: PyPDF, pdf2image
- **Image Analysis**: Gemini Vision API

## 📋 Prerequisites

- Python 3.8+
- Google Gemini API Key ([Get it here](https://makersuite.google.com/app/apikey))
- Chrome/Chromium browser (for web scraping)
- ChromeDriver (for web scraping) - [Download here](https://chromedriver.chromium.org/)

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## 📚 Usage

### Step 1: Ingest Documentation

Choose one or both methods to populate the vector database:

#### Option A: From PDF Documents

1. Place your PDF files in the `docs/` folder
2. Run the ingestion script:
   ```bash
   python ingest.py
   ```

#### Option B: From Web (GeneXus Documentation)

1. Ensure ChromeDriver is installed and accessible
2. Run the web scraping ingestion:
   ```bash
   python ingest_site.py
   ```

#### Option C: Process Images from PDFs

To extract and describe images using Gemini Vision:
```bash
python image_processor.py
```

### Step 2: Verify the Index

Check if the vector database was created successfully:
```bash
python check_index.py
```

### Step 3: Run the Assistant

Start the Streamlit application:
```bash
streamlit run app.py
```

The assistant will be available at `http://localhost:8501`

## 🔧 Configuration

All configuration can be done through the `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Your Gemini API key | *Required* |
| `CHROMA_DB_PATH` | Vector database location | `./chroma_db` |
| `DOCS_PATH` | PDF documents folder | `./docs` |
| `CHUNK_SIZE` | Text chunk size for splitting | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `RETRIEVAL_K` | Number of chunks to retrieve | `3` |
| `MAX_ARTICLES_TO_INDEX` | Max articles for web scraping | `50` |
| `MAX_PAGES_TO_SCAN` | Max pages to scan during scraping | `10` |
| `CHROME_DRIVER_PATH` | Path to ChromeDriver (optional) | System PATH |

## 📁 Project Structure

```
.
├── app.py                 # Main Streamlit application
├── ingest.py             # PDF document ingestion
├── ingest_site.py        # Web scraping ingestion
├── image_processor.py    # Image extraction and description
├── check_index.py        # Verify vector database
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── .gitignore           # Git ignore rules
├── README.md            # This file
├── docs/                # Place PDF files here
├── chroma_db/           # Vector database (auto-generated)
└── processed_text/      # Enriched text from images (auto-generated)
```

## 🎯 How It Works

1. **Document Ingestion**: PDFs or web pages are loaded and split into chunks
2. **Embedding Creation**: Each chunk is converted to a vector using Google's text-embedding-004 model
3. **Vector Storage**: Embeddings are stored in ChromaDB for efficient similarity search
4. **Query Processing**: User questions are embedded and matched against stored documents
5. **Response Generation**: Relevant context is passed to Gemini 2.0 Flash to generate answers

## 🔍 Features Explained

### RAG (Retrieval Augmented Generation)

The assistant uses RAG to provide accurate, context-aware responses:
- Searches the vector database for relevant documentation
- Retrieves the top K most similar chunks
- Passes them as context to the LLM
- Generates responses based on actual documentation

### Web Scraping

The web scraper (`ingest_site.py`):
- Uses Selenium for dynamic content handling
- Implements efficient waiting strategies (reduced from 10s to 2s)
- Extracts articles from GeneXus 18 documentation
- Filters out non-article content (images, CSS, JS)
- Supports pagination with configurable limits

### Image Processing

The image processor (`image_processor.py`):
- Converts PDF pages to images
- Uses Gemini Vision API to describe technical content
- Focuses on code, diagrams, and object properties
- Saves enriched text for later ingestion

## 🐛 Troubleshooting

### "GEMINI_API_KEY not configured"
- Make sure you've created a `.env` file from `.env.example`
- Verify your API key is correct and active

### "Error loading the database"
- Run `python ingest.py` or `python ingest_site.py` first
- Check if `chroma_db/` folder exists

### "ChromeDriver not found"
- Install ChromeDriver matching your Chrome version
- Set `CHROME_DRIVER_PATH` in `.env` or add to system PATH

### "No PDF files found"
- Place PDF files in the `docs/` folder
- Ensure files have `.pdf` extension

## 🔒 Security Notes

- Never commit your `.env` file with real API keys
- Keep your `GEMINI_API_KEY` secret
- The `.gitignore` file is configured to exclude sensitive files

## 📝 Improvements Made

This improved version includes:

✅ **Configuration Management**
- Environment variables for all settings
- `.env.example` template file
- Removed hardcoded paths

✅ **Code Quality**
- Better error handling throughout
- Input validation
- Removed unused code
- Updated deprecated methods
- Added comprehensive comments

✅ **Performance**
- Reduced web scraper wait times (10s → 2s)
- Efficient WebDriver waiting strategies
- Headless browser mode for scraping

✅ **User Experience**
- Clear error messages with emojis
- Helpful troubleshooting hints
- Progress indicators
- Better feedback during operations

✅ **Project Structure**
- Complete requirements.txt
- Proper .gitignore
- Auto-created directories
- Comprehensive documentation

✅ **Security**
- No hardcoded credentials
- Environment-based configuration
- Protected sensitive files

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows existing style
- All dependencies are in requirements.txt
- New features are documented
- Tests pass (if applicable)

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- GeneXus documentation team
- Google Gemini AI
- LangChain framework
- ChromaDB vector database

---

**Need help?** Check the troubleshooting section or open an issue.
