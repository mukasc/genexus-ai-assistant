# 📚 New Ingestion System - User Guide

## 🎉 What's New

Your GeneXus AI Assistant now has a **completely redesigned ingestion system**:

✅ **Upload PDF files directly** - No need to place files in folders  
✅ **Enter URLs manually** - Precise control over what to ingest  
✅ **Instant processing** - See results immediately  
✅ **Multiple file support** - Upload several PDFs at once  

---

## 🚀 How to Use

### Option 1: 📄 Upload PDF Files

**Perfect for**: Documentation, manuals, guides, tutorials

**Steps**:

1. **Click "📚 Ingest Documents"** in the sidebar

2. **Select "📄 Upload PDF Files"**

3. **Choose your files**:
   - A file dialog will open
   - Select one or multiple PDF files
   - Click "Open"

4. **Wait for processing** (~10-30 seconds per file):
   - Progress shown in sidebar
   - "✅ Success" message when done
   - Shows number of chunks created

5. **Start chatting!** 
   - Database status updates automatically
   - New documents are now searchable

**Example**:
```
1. Click "📚 Ingest Documents"
2. Click "📄 Upload PDF Files"
3. Select: genexus_manual.pdf, tutorial.pdf
4. Wait for: "✅ Successfully ingested 2 PDF file(s)"
5. Ask: "What is covered in the documentation?"
```

---

### Option 2: 🌐 Ingest from URL

**Perfect for**: Online documentation, specific web pages, articles

**Steps**:

1. **Click "📚 Ingest Documents"** in the sidebar

2. **Select "🌐 From URL"**

3. **Enter the URL**:
   - Paste the full URL
   - Example: `https://docs.genexus.com/en/wiki?12345`
   - Must start with `http://` or `https://`

4. **Click "✅ Ingest"**:
   - Processing starts immediately
   - Progress shown in sidebar

5. **Wait for completion** (~5-15 seconds):
   - "✅ Success" message when done
   - Shows number of chunks created

6. **Start chatting!**

**Example URLs**:
```
https://docs.genexus.com/en/wiki?12345,GeneXus+18
https://wiki.genexus.com/commwiki/servlet/wiki?12345
https://training.genexus.com/en/learning-path
```

**Pro Tips**:
- ✅ Works with any public webpage
- ✅ Best with documentation pages
- ✅ Can ingest multiple URLs one by one
- ✅ New content adds to existing index

---

## 🔧 Technical Details

### What Happens During Ingestion?

**PDF Upload**:
1. Files uploaded to backend
2. Text extracted from each page
3. Content split into searchable chunks
4. Embeddings created using Gemini AI
5. Stored in ChromaDB vector database
6. RAG system reinitialized

**URL Ingestion**:
1. Web page content fetched
2. HTML parsed and cleaned
3. Text extracted from page
4. Content split into chunks
5. Embeddings created
6. Added to vector database
7. RAG system reinitialized

### Processing Time

| Source | Size | Time | Chunks |
|--------|------|------|--------|
| Small PDF (10-50 pages) | 1 file | 10-20 sec | 10-50 |
| Large PDF (100+ pages) | 1 file | 30-60 sec | 100-200 |
| Web Page | Single URL | 5-15 sec | 10-50 |
| Multiple PDFs | 5 files | 1-2 min | 50-250 |

---

## ✅ Status Indicators

### In the Sidebar:

**Before Any Ingestion:**
```
📊 Status
⚠️ Degraded
API Key: ✅
Database: ❌

Message: "Vector database not loaded. Run ingestion scripts."
```

**During Ingestion:**
```
⏳ Uploading and processing 2 PDF file(s)...
```
or
```
⏳ Processing URL: https://docs.genexus.com/...
```

**After Successful Ingestion:**
```
✅ Successfully ingested 2 PDF file(s). Created 45 chunks.

📊 Status
✅ Online
API Key: ✅
Database: ✅

📊 Index
45 chunks

Message: "System operational"
```

---

## 💡 Best Practices

### For PDF Uploads:
- ✅ **Use clear, text-based PDFs** (not scanned images)
- ✅ **Upload in batches** (5-10 files at a time)
- ✅ **Use relevant names** (helps you track what's ingested)
- ✅ **Check success message** before uploading more

### For URL Ingestion:
- ✅ **Use documentation pages** (structured content works best)
- ✅ **Verify URL accessibility** (must be publicly accessible)
- ✅ **One URL at a time** (for better tracking)
- ✅ **Wait for completion** before ingesting next URL

### General Tips:
- 📏 **Quality over quantity** - Better docs = Better answers
- 🔄 **Incremental ingestion** - Add more anytime
- 💾 **Database persists** - Documents stay indexed
- 🎯 **Test with questions** - Verify ingestion worked

---

## 🐛 Troubleshooting

### "Network Error" when uploading

**Cause**: Backend not responding or CORS issue

**Solution**:
```bash
# Check backend status
curl http://localhost:8001/api/health

# Restart backend if needed
sudo supervisorctl restart backend

# Wait 5 seconds and try again
```

### "No valid PDF documents found"

**Cause**: Files are not PDFs or are corrupted

**Solution**:
- Verify files have `.pdf` extension
- Try opening PDFs locally first
- Use different PDF files

### "Error during ingestion" for URLs

**Cause**: URL not accessible or invalid content

**Solution**:
- Verify URL opens in browser
- Check URL starts with `http://` or `https://`
- Try a different page
- Check if site requires authentication

### "API key not configured"

**Cause**: API key missing or not loaded

**Solution**:
```bash
# Verify API key in .env
cat /app/.env | grep GEMINI_API_KEY

# Restart backend
sudo supervisorctl restart backend
```

### Files uploaded but status not updating

**Cause**: Frontend needs refresh

**Solution**:
- Refresh the webpage
- Check sidebar for updated chunk count
- Try asking a question to verify

---

## 📊 Monitoring Ingestion

### Check What's Indexed:

**Via API**:
```bash
curl http://localhost:8001/api/index-status
```

**Via Command Line**:
```bash
python /app/check_index.py
```

**Via UI**:
- Look at sidebar "📊 Index" section
- Shows total chunk count

### View Backend Logs:

```bash
# Real-time logs
tail -f /var/log/supervisor/backend.err.log

# Recent logs
tail -50 /var/log/supervisor/backend.err.log
```

---

## 🎯 Example Workflows

### Workflow 1: Quick Start with URLs

```
1. Open the app
2. Click "📚 Ingest Documents"
3. Select "🌐 From URL"
4. Paste: https://docs.genexus.com/en/wiki?12345
5. Click "✅ Ingest"
6. Wait for success message
7. Repeat for 2-3 more important pages
8. Start asking questions!
```

### Workflow 2: Custom Documentation

```
1. Prepare your GeneXus PDFs
2. Open the app
3. Click "📚 Ingest Documents"
4. Select "📄 Upload PDF Files"
5. Select all your PDFs (Ctrl+Click or Cmd+Click)
6. Click "Open"
7. Wait for processing
8. Verify success message
9. Ask specific questions about your docs
```

### Workflow 3: Mixed Sources

```
1. Upload your custom PDFs first
2. Then add official documentation URLs
3. Combine internal and external knowledge
4. Get comprehensive answers!
```

---

## 🔄 Re-Ingestion

### Adding More Documents

**Good news**: Ingestion is **incremental**!

- Upload new PDFs anytime
- Add more URLs anytime
- Old documents remain indexed
- No need to start over

### Updating Existing Content

If a document has changed:
1. Delete old database: `rm -rf /app/chroma_db`
2. Re-upload all documents
3. Or just add the updated version

---

## 🎉 Success Checklist

Before asking questions, verify:

- ✅ Status shows "✅ Online"
- ✅ Database shows "✅"
- ✅ Index shows chunk count > 0
- ✅ Success message received
- ✅ No error messages in sidebar

If all green, you're ready to chat!

---

## 📞 Support

**Check documentation:**
- `/app/README_FULLSTACK.md` - Complete technical docs
- `/app/FINAL_SETUP_STATUS.md` - Setup status
- `/app/GETTING_STARTED.md` - Quick start guide

**Check logs:**
```bash
tail -f /var/log/supervisor/backend.err.log
```

**Restart services:**
```bash
sudo supervisorctl restart all
```

---

**Enjoy your enhanced GeneXus AI Assistant! 🚀**
