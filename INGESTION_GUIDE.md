# 📚 Document Ingestion Guide

Your GeneXus AI Assistant now has **built-in ingestion options** directly in the UI!

## 🎯 Two Ways to Ingest Documents

### Option 1: 📄 From PDF Files

**Best for**: Custom documentation, manuals, or specific PDF resources

**Steps**:
1. **Add PDFs to the docs folder**:
   ```bash
   cp your_genexus_manual.pdf /app/docs/
   ```

2. **Click "Ingest Documents" in the UI sidebar**

3. **Select "📄 From PDF Files"**

4. **Wait for completion** (~30 seconds to 2 minutes depending on file size)

5. **Start asking questions!**

**What it does**:
- Reads all PDF files from `/app/docs/`
- Extracts text from each page
- Splits into chunks for efficient retrieval
- Creates embeddings using Gemini
- Stores in ChromaDB vector database

---

### Option 2: 🌐 From GeneXus Website

**Best for**: Official GeneXus documentation

**Steps**:
1. **Click "Ingest Documents" in the UI sidebar**

2. **Select "🌐 From GeneXus Website"**

3. **Wait for completion** (~5-15 minutes)
   - Scrapes up to 50 articles by default
   - Shows progress in the sidebar

4. **Start asking questions!**

**What it does**:
- Opens GeneXus documentation website
- Scrapes articles about GeneXus 18
- Extracts relevant content
- Creates embeddings and indexes them
- Combines with any existing documentation

**Configuration** (optional):
Edit `/app/.env` to adjust:
```env
MAX_ARTICLES_TO_INDEX=50    # Number of articles to scrape
MAX_PAGES_TO_SCAN=10        # Number of search pages to scan
```

---

## 📊 Checking Ingestion Status

### In the UI:
- Look at the sidebar "📊 Index" section
- Shows number of document chunks indexed
- Green indicator when database is loaded

### Via API:
```bash
curl http://localhost:8001/api/index-status
```

### Via Command Line:
```bash
python /app/check_index.py
```

---

## 🔄 Re-Ingesting or Adding More Documents

**Good news**: Ingestion is **incremental**!

### To add more PDFs:
1. Add new PDFs to `/app/docs/`
2. Click "📄 From PDF Files" again
3. New documents are added to existing index

### To update web documentation:
1. Click "🌐 From GeneXus Website" again
2. New articles are added (duplicates are handled)

### To start fresh:
```bash
# Delete the vector database
rm -rf /app/chroma_db

# Re-ingest from scratch
# Use the UI buttons or:
python /app/ingest.py
# or
python /app/ingest_site.py
```

---

## 🐛 Troubleshooting

### "No PDF files found"
**Solution**: 
```bash
# Check if docs folder has PDFs
ls -la /app/docs/

# Add some PDFs
cp your_file.pdf /app/docs/
```

### "Web scraping failed"
**Solution**:
```bash
# Check if chromium is installed
which chromium-browser

# If not, install it
sudo apt-get update
sudo apt-get install -y chromium chromium-driver
```

### "Ingestion stuck"
**Solution**:
```bash
# Check backend logs
tail -f /var/log/supervisor/backend.err.log

# Restart backend if needed
sudo supervisorctl restart backend
```

### "Index not updating"
**Solution**:
1. Wait a few minutes for ingestion to complete
2. Refresh the page
3. Check index status: `curl http://localhost:8001/api/index-status`

---

## 💡 Tips & Best Practices

### For PDF Ingestion:
- ✅ Use clear, text-based PDFs (not scanned images)
- ✅ Organize PDFs by topic in docs/ folder
- ✅ File names don't matter (content is indexed)
- ✅ Can mix multiple languages

### For Web Scraping:
- ✅ Run during off-peak hours (faster)
- ✅ First run takes longer (subsequent runs are cached)
- ✅ Articles are deduplicated automatically
- ✅ Can run multiple times safely

### General:
- 📏 Larger documents = longer ingestion time
- 🔄 Re-run ingestion when documentation updates
- 💾 Vector database persists between restarts
- 🎯 Quality documents = better answers

---

## 📈 Expected Performance

| Source | Documents | Time | Chunks Created |
|--------|-----------|------|----------------|
| Small PDF (50 pages) | 1 | 30 sec | ~50-100 |
| Large PDF (500 pages) | 1 | 2-3 min | ~500-800 |
| Web (50 articles) | 50 | 5-15 min | ~500-1000 |

---

## 🎉 You're All Set!

Once ingestion completes:
- ✅ Green "Database: ✅" in sidebar
- ✅ Document count shows in "Index" section
- ✅ Chat responds with documentation-based answers
- ✅ Assistant becomes GeneXus-specialized!

**Try asking**:
- "What is a GeneXus Transaction?"
- "How do I create a Data Provider?"
- "Explain GeneXus Web Panels"

---

**Need more help?** Check `/app/README_FULLSTACK.md` for complete documentation.
