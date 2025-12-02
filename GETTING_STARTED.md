# 🚀 Getting Started with GeneXus AI Assistant

## ✅ Current Status

Your application is **RUNNING** but needs configuration to work fully.

### What's Working ✅
- ✅ Backend API (FastAPI) - Running on port 8001
- ✅ Frontend (React) - Running on port 3000  
- ✅ System architecture is set up correctly

### What Needs Configuration ⚠️
- ⚠️ Gemini API Key (required)
- ⚠️ Document ingestion (optional but recommended)

## 🎯 3 Steps to Get It Working

### Step 1: Add Your API Key (REQUIRED)

1. **Get your Gemini API key:**
   - Go to: https://makersuite.google.com/app/apikey
   - Create or copy your API key

2. **Add it to the `.env` file:**
   ```bash
   # Edit the .env file
   nano /app/.env
   ```
   
   Add this line:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

3. **Restart the backend:**
   ```bash
   sudo supervisorctl restart backend
   ```

### Step 2: Add Documentation (RECOMMENDED)

The AI assistant needs documentation to answer questions. Choose one option:

**Option A: Use PDF Documents**
```bash
# 1. Add PDF files to the docs folder
cp your_genexus_docs.pdf /app/docs/

# 2. Run ingestion
python /app/ingest.py
```

**Option B: Scrape GeneXus Website**
```bash
# This will download and index GeneXus documentation
python /app/ingest_site.py
```

> **Note**: Web scraping takes 5-15 minutes and downloads 50 articles by default.

### Step 3: Test the Application

1. **Check backend health:**
   ```bash
   curl http://localhost:8001/api/health
   ```
   
   Should show:
   ```json
   {
     "status": "healthy",
     "api_key_configured": true,
     "database_loaded": true,
     "message": "System operational"
   }
   ```

2. **Open the frontend:**
   - Your app should be accessible via the preview URL
   - Or visit: http://localhost:3000

3. **Try asking a question:**
   - "What is GeneXus?"
   - "Como criar um Data Provider?"
   - "Explain GeneXus transactions"

## 📊 Quick Commands Reference

### Check Status
```bash
# View all services
sudo supervisorctl status

# Check backend logs
tail -f /var/log/supervisor/backend.err.log

# Check frontend logs  
tail -f /var/log/supervisor/frontend.err.log
```

### Restart Services
```bash
# Restart backend only
sudo supervisorctl restart backend

# Restart frontend only
sudo supervisorctl restart frontend

# Restart everything
sudo supervisorctl restart all
```

### Test API
```bash
# Health check
curl http://localhost:8001/api/health

# Index status
curl http://localhost:8001/api/index-status

# Test chat (requires API key + indexed docs)
curl -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

## 🎨 What You'll See

### Before Configuration:
- Frontend loads but shows warnings
- System status shows "degraded"
- Chat doesn't work yet

### After Configuration:
- ✅ Green status indicators
- ✅ Document count displayed
- ✅ Chat responds to questions
- ✅ Full functionality

## 🐛 Common Issues

### "API key not configured"
**Solution**: Add `GEMINI_API_KEY` to `/app/.env` and restart backend

### "Vector database not loaded"
**Solution**: Run `python /app/ingest.py` or `python /app/ingest_site.py`

### "Cannot connect to backend"
**Solution**: 
```bash
# Check if backend is running
sudo supervisorctl status backend

# If not, restart it
sudo supervisorctl restart backend
```

### "Chat returns errors"
**Solution**: Make sure both API key AND documentation are configured

## 📚 Next Steps

Once everything is working:

1. **Explore the documentation files:**
   - See: `/app/README_FULLSTACK.md` for full documentation
   - See: `/app/IMPROVEMENTS_SUMMARY.md` for all improvements made

2. **Customize the assistant:**
   - Edit prompts in `/app/backend/server.py`
   - Modify UI in `/app/frontend/src/App.js`
   - Adjust configuration in `/app/.env`

3. **Add more documentation:**
   - Add PDFs to `/app/docs/`
   - Run `python /app/ingest.py` again
   - Documents are added incrementally

4. **Monitor performance:**
   - Check logs for errors
   - Monitor response times
   - Adjust RETRIEVAL_K for better results

## ✨ Quick Start Script

Want to do everything at once? Run:

```bash
cd /app
python setup.py
```

This interactive script will:
- Check your configuration
- Validate dependencies
- Guide you through setup
- Test the system

## 🎉 That's It!

Once you complete Step 1 (API key) and Step 2 (documentation), your GeneXus AI Assistant will be fully functional!

Need help? Check the logs or see the full documentation in `/app/README_FULLSTACK.md`.

---

**Happy coding! 🚀**
