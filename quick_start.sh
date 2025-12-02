#!/bin/bash

# GeneXus AI Assistant - Quick Start Script
# This script helps you get started quickly

echo "======================================================================"
echo "  🤖 GeneXus AI Assistant - Quick Start"
echo "======================================================================"
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found"
    echo "📝 Creating .env from .env.example..."
    
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Created .env file"
        echo ""
        echo "⚠️  IMPORTANT: Edit .env and add your GEMINI_API_KEY"
        echo "   Get your API key from: https://makersuite.google.com/app/apikey"
        echo ""
        read -p "Press Enter after you've added your API key..."
    else
        echo "❌ .env.example not found!"
        exit 1
    fi
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed"
else
    echo "❌ Error installing dependencies"
    exit 1
fi

# Run setup validation
echo ""
echo "🔍 Running setup validation..."
python setup.py

# Ask user what they want to do
echo ""
echo "======================================================================"
echo "  What would you like to do?"
echo "======================================================================"
echo ""
echo "  1. Ingest PDF documents (from docs/ folder)"
echo "  2. Scrape GeneXus web documentation"
echo "  3. Check vector database index"
echo "  4. Start the Streamlit app"
echo "  5. Exit"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting PDF ingestion..."
        python ingest.py
        ;;
    2)
        echo ""
        echo "🚀 Starting web scraping ingestion..."
        echo "⚠️  This may take several minutes..."
        python ingest_site.py
        ;;
    3)
        echo ""
        echo "🔍 Checking vector database..."
        python check_index.py
        ;;
    4)
        echo ""
        echo "🚀 Starting Streamlit app..."
        echo "💡 The app will open in your browser"
        echo "💡 Press Ctrl+C to stop"
        echo ""
        streamlit run app.py
        ;;
    5)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "======================================================================"
echo "  ✅ Done!"
echo "======================================================================"
echo ""
