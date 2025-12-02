#!/usr/bin/env python3
"""
Setup utility for GeneXus AI Assistant
Helps users configure the environment and verify installation
"""

import os
import sys
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_step(number, text):
    """Print a step number"""
    print(f"\n{number}. {text}")

def check_file_exists(filepath):
    """Check if a file exists"""
    return os.path.exists(filepath)

def create_env_file():
    """Create .env file from .env.example"""
    if check_file_exists(".env"):
        print("   ✅ .env file already exists")
        return True
    
    if not check_file_exists(".env.example"):
        print("   ❌ .env.example not found")
        return False
    
    try:
        with open(".env.example", "r") as src:
            content = src.read()
        
        with open(".env", "w") as dst:
            dst.write(content)
        
        print("   ✅ Created .env file from .env.example")
        print("   ⚠️  Please edit .env and add your GEMINI_API_KEY")
        return True
    except Exception as e:
        print(f"   ❌ Error creating .env: {e}")
        return False

def check_directories():
    """Check and create necessary directories"""
    directories = ["docs", "chroma_db", "processed_text"]
    
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"   ✅ Created directory: {directory}/")
            except Exception as e:
                print(f"   ❌ Error creating {directory}/: {e}")
        else:
            print(f"   ✅ Directory exists: {directory}/")

def check_dependencies():
    """Check if key dependencies are installed"""
    dependencies = {
        "streamlit": "Streamlit",
        "langchain": "LangChain",
        "langchain_google_genai": "LangChain Google GenAI",
        "chromadb": "ChromaDB",
        "dotenv": "python-dotenv",
        "selenium": "Selenium",
    }
    
    missing = []
    
    for package, name in dependencies.items():
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - NOT INSTALLED")
            missing.append(name)
    
    if missing:
        print(f"\n   ⚠️  Missing packages: {', '.join(missing)}")
        print("   💡 Run: pip install -r requirements.txt")
        return False
    
    return True

def check_api_key():
    """Check if API key is configured"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key or api_key == "your_gemini_api_key_here":
            print("   ❌ GEMINI_API_KEY not configured in .env")
            print("   💡 Get your API key from: https://makersuite.google.com/app/apikey")
            return False
        
        print("   ✅ GEMINI_API_KEY is configured")
        return True
        
    except Exception as e:
        print(f"   ❌ Error checking API key: {e}")
        return False

def check_chrome_driver():
    """Check if Chrome/ChromeDriver is available"""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        
        try:
            driver = webdriver.Chrome(options=options)
            driver.quit()
            print("   ✅ Chrome/ChromeDriver available")
            return True
        except Exception:
            print("   ⚠️  Chrome/ChromeDriver not found (only needed for web scraping)")
            print("   💡 Download from: https://chromedriver.chromium.org/")
            return False
            
    except ImportError:
        print("   ⚠️  Selenium not installed")
        return False

def main():
    """Main setup function"""
    print_header("🤖 GeneXus AI Assistant - Setup Utility")
    
    print("\nThis utility will help you set up the GeneXus AI Assistant.")
    
    # Step 1: Check Python version
    print_step(1, "Checking Python version")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ❌ Python 3.8+ required")
        return
    else:
        print("   ✅ Python version OK")
    
    # Step 2: Create .env file
    print_step(2, "Setting up environment configuration")
    env_created = create_env_file()
    
    # Step 3: Check/create directories
    print_step(3, "Checking directories")
    check_directories()
    
    # Step 4: Check dependencies
    print_step(4, "Checking Python dependencies")
    deps_ok = check_dependencies()
    
    # Step 5: Check API key
    print_step(5, "Checking Gemini API key")
    api_ok = check_api_key()
    
    # Step 6: Check ChromeDriver (optional)
    print_step(6, "Checking Chrome/ChromeDriver (optional)")
    check_chrome_driver()
    
    # Summary
    print_header("📋 Setup Summary")
    
    if deps_ok and api_ok:
        print("\n✅ Setup complete! You're ready to go.")
        print("\n📚 Next steps:")
        print("   1. Add PDF files to the 'docs/' folder (optional)")
        print("   2. Run: python ingest.py (to index PDFs)")
        print("   3. Run: python ingest_site.py (to scrape web docs)")
        print("   4. Run: streamlit run app.py (to start the assistant)")
        print("\n💡 For more information, check README_IMPROVED.md")
    else:
        print("\n⚠️  Setup incomplete. Please address the issues above.")
        
        if not deps_ok:
            print("\n   📦 Install dependencies:")
            print("      pip install -r requirements.txt")
        
        if not api_ok:
            print("\n   🔑 Configure API key:")
            print("      1. Edit .env file")
            print("      2. Add your GEMINI_API_KEY")
            print("      3. Get key from: https://makersuite.google.com/app/apikey")
    
    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    main()
