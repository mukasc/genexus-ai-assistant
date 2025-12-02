# 🤖 GeneXus AI Assistant - Full Stack Version

> **Note**: This is the full-stack version (FastAPI + React) adapted for deployment environments.  
> For the original Streamlit version, see the Python files in the root directory.

## 🌟 Architecture

This application uses a modern full-stack architecture:

- **Backend**: FastAPI (Python) - Provides REST API for RAG functionality
- **Frontend**: React - Interactive chat interface
- **Vector Store**: ChromaDB - Stores document embeddings
- **LLM**: Google Gemini 2.0 Flash - Generates responses

## 📁 Project Structure

```
/app/
├── backend/
│   ├── server.py              # FastAPI application
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # Environment config (links to root)
├── frontend/
│   ├── src/
│   │   ├── App.js            # Main React component
│   │   ├── App.css           # Styles
│   │   ├── index.js          # Entry point
│   │   └── index.css         # Global styles
│   ├── public/
│   │   └── index.html        # HTML template
│   ├── package.json          # Node dependencies
│   └── .env                  # Frontend config
├── docs/                      # PDF documents for ingestion
├── chroma_db/                 # Vector database (generated)
├── .env                       # Main configuration file
├── ingest.py                  # PDF ingestion script
├── ingest_site.py             # Web scraping script
└── README_FULLSTACK.md        # This file
```

## 🚀 Quick Start

### Prerequisites

1. **API Key**: Get your Gemini API key from https://makersuite.google.com/app/apikey
2. **Dependencies**: Python 3.8+ and Node.js 14+

### Setup

1. **Configure Environment**:
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env and add your GEMINI_API_KEY
   nano .env
   ```

2. **Ingest Documentation** (Choose one or both):
   
   **Option A - From PDFs:**
   ```bash
   # Place PDF files in docs/ folder
   python ingest.py
   ```
   
   **Option B - From Web:**
   ```bash
   python ingest_site.py
   ```

3. **Start Services** (if not using supervisor):
   ```bash
   # Backend (Terminal 1)
   cd backend
   pip install -r requirements.txt
   uvicorn server:app --host 0.0.0.0 --port 8001 --reload
   
   # Frontend (Terminal 2)
   cd frontend
   yarn install
   yarn start
   ```

### With Supervisor (Production)

Services are managed by supervisor and start automatically:

```bash
# Check status
sudo supervisorctl status

# Restart services
sudo supervisorctl restart backend
sudo supervisorctl restart frontend
sudo supervisorctl restart all

# View logs
tail -f /var/log/supervisor/backend.err.log
tail -f /var/log/supervisor/frontend.err.log
```

## 📡 API Endpoints

### Backend API (Port 8001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/` | GET | API information |
| `/api/health` | GET | System health check |
| `/api/index-status` | GET | Vector database status |
| `/api/chat` | POST | Chat with the assistant |

### Example API Calls

**Health Check:**
```bash
curl http://localhost:8001/api/health
```

**Chat:**
```bash
curl -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is GeneXus?"}'
```

## 🎨 Frontend

The React frontend provides:

- ✅ Interactive chat interface
- ✅ Real-time system status
- ✅ Message history
- ✅ Loading indicators
- ✅ Error handling
- ✅ Responsive design

Access at: `http://localhost:3000`

## ⚙️ Configuration

All configuration is done through the root `.env` file:

```env
# Required
GEMINI_API_KEY=your_api_key_here

# Optional (with defaults)
CHROMA_DB_PATH=./chroma_db
DOCS_PATH=./docs
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=3
```

## 🔧 Development

### Backend Development

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn server:app --reload --port 8001

# Run tests
pytest
```

### Frontend Development

```bash
cd frontend

# Install dependencies
yarn install

# Start development server
yarn start

# Build for production
yarn build
```

## 🐛 Troubleshooting

### Backend Issues

**"API key not configured"**
- Edit `.env` file in root directory
- Add your `GEMINI_API_KEY`
- Restart backend: `sudo supervisorctl restart backend`

**"Vector database not loaded"**
- Run `python ingest.py` or `python ingest_site.py`
- Check if `chroma_db/` directory exists
- Verify documents were processed successfully

**Backend won't start**
```bash
# Check logs
tail -50 /var/log/supervisor/backend.err.log

# Check if port is in use
lsof -i :8001

# Restart backend
sudo supervisorctl restart backend
```

### Frontend Issues

**"Cannot connect to backend"**
- Check if backend is running: `curl http://localhost:8001/api/health`
- Verify `REACT_APP_BACKEND_URL` in `frontend/.env`
- Check CORS settings in backend

**Frontend won't start**
```bash
# Check logs
tail -50 /var/log/supervisor/frontend.err.log

# Clear cache and reinstall
cd frontend
rm -rf node_modules package-lock.json
yarn install

# Restart frontend
sudo supervisorctl restart frontend
```

## 📊 System Status

Check system status through the API or frontend sidebar:

- **API Key**: Shows if Gemini API key is configured
- **Database**: Shows if vector database is loaded
- **Index**: Shows number of document chunks indexed

## 🔒 Security

- **Environment Variables**: Never commit `.env` file
- **API Keys**: Keep your Gemini API key secret
- **CORS**: Backend configured to accept requests from frontend
- **Input Validation**: All user inputs are validated

## 📈 Performance

- **Response Time**: Typically 2-5 seconds per query
- **Concurrent Users**: Supports multiple simultaneous users
- **Caching**: LLM responses are not cached (always fresh)
- **Rate Limiting**: Limited by Gemini API quotas

## 🔄 Updating

### Update Backend
```bash
cd backend
pip install -r requirements.txt --upgrade
sudo supervisorctl restart backend
```

### Update Frontend
```bash
cd frontend
yarn upgrade
sudo supervisorctl restart frontend
```

### Update Documentation Index
```bash
# Re-run ingestion scripts
python ingest.py
python ingest_site.py
```

## 📚 Additional Scripts

All original Python scripts are still available in the root directory:

- `app.py` - Original Streamlit version
- `ingest.py` - PDF ingestion
- `ingest_site.py` - Web scraping
- `image_processor.py` - Image description
- `check_index.py` - Verify vector database
- `setup.py` - Setup wizard
- `validate_improvements.py` - Validation script

## 🎯 Production Deployment

For production deployment:

1. ✅ Set `NODE_ENV=production` in frontend
2. ✅ Build frontend: `cd frontend && yarn build`
3. ✅ Use production ASGI server (Gunicorn + Uvicorn)
4. ✅ Set up reverse proxy (Nginx)
5. ✅ Enable HTTPS
6. ✅ Configure rate limiting
7. ✅ Set up monitoring and logging

## 🤝 Support

- Backend API: http://localhost:8001/api/
- Frontend: http://localhost:3000
- Logs: `/var/log/supervisor/`

## 📄 License

[Add your license here]

---

**Built with ❤️ using FastAPI, React, LangChain, and Google Gemini**
