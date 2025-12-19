import os
from dotenv import load_dotenv
from google import genai
from PIL import Image
from io import BytesIO
from pdf2image import convert_from_path

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found. Check your .env file.")

# Initialize Gemini client
try:
    gemini_client = genai.Client(api_key=API_KEY)
except Exception as e:
    raise ValueError(f"❌ Error initializing Gemini client: {e}")

def describe_image_with_gemini(image_path_or_bytes):
    """Send an image to Gemini Vision to generate a detailed description."""
    
    # Engineering prompt to get a technical and useful description for RAG
    prompt = (
        "Describe this image in a technical and concise way for a GeneXus developer. "
        "Focus on elements like object names, attributes, data flow diagrams, "
        "properties or visible code. Start the description with '[IMAGE DESCRIBED]: '."
    )
    
    # If it's a path, open the image; otherwise, assume it's bytes
    try:
        if isinstance(image_path_or_bytes, str):
            img = Image.open(image_path_or_bytes)
        else:
            img = Image.open(image_path_or_bytes)
    except Exception as e:
        print(f"❌ Error opening image: {e}")
        return "[IMAGE NOT DESCRIBED DUE TO ERROR OPENING FILE]"
    
    try:
        # Use Pro Vision model for description
        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash-exp',  # Gemini 2.0 Flash is multimodal and faster
            contents=[prompt, img]
        )
        return response.text
    except Exception as e:
        print(f"❌ Error describing image with Gemini: {e}")
        return "[IMAGE NOT DESCRIBED DUE TO API ERROR]"


def extract_and_describe_from_pdf(pdf_path, output_dir="./processed_text"):
    """Simulate text extraction and image description for enrichment."""
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return None
    
    print(f"📝 Processing {pdf_path}...")
    
    # This is a simplification. Real extraction in a complex PDF is difficult.
    # Here, we convert each page to an image and describe that image.
    
    try:
        pages = convert_from_path(pdf_path)
    except Exception as e:
        print(f"❌ Error converting PDF to images: {e}")
        return None
    
    enriched_text = ""
    total_pages = len(pages)
    
    print(f"📊 Found {total_pages} page(s) in PDF")
    
    for i, page_image in enumerate(pages, 1):
        print(f"  Processing page {i}/{total_pages}...")
        
        # 1. Get page text (simulation/improvement needed for real text)
        # The ideal method here would be to use an advanced OCR library to get the layout
        # But for the prototype, we focus only on visual description.
        
        # 2. Describe the Page Image (slow but effective)
        try:
            with BytesIO() as output:
                page_image.save(output, format="PNG")
                image_bytes = output.getvalue()
            
            # Send the page (as image) to Gemini
            description = describe_image_with_gemini(BytesIO(image_bytes))
            
            # 3. Add description to enriched text
            enriched_text += f"\n\n--- START OF VISUAL CONTENT PAGE {i} ---\n{description}\n--- END OF VISUAL CONTENT ---\n\n"
        except Exception as e:
            print(f"  ⚠️ Error processing page {i}: {e}")
            enriched_text += f"\n\n--- PAGE {i} - ERROR PROCESSING ---\n\n"
    
    # Save enriched text (with descriptions) to a .txt file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ Created directory: {output_dir}")
    
    output_filename = os.path.join(
        output_dir,
        os.path.basename(pdf_path).replace(".pdf", "_enriched.txt")
    )
    
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(enriched_text)
        print(f"✅ Enriched content saved to: {output_filename}")
        return output_filename
    except Exception as e:
        print(f"❌ Error saving enriched text: {e}")
        return None

# Example execution (for testing)
if __name__ == "__main__":
    # Assuming you have a test PDF in the docs folder
    docs_path = os.getenv("DOCS_PATH", "./docs")
    
    # Look for PDF files
    if os.path.exists(docs_path):
        pdf_files = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]
        
        if pdf_files:
            print(f"\n📂 Found {len(pdf_files)} PDF file(s) in {docs_path}")
            print("\nProcessing first PDF as test...")
            pdf_to_process = os.path.join(docs_path, pdf_files[0])
            extract_and_describe_from_pdf(pdf_to_process)
        else:
            print(f"⚠️ No PDF files found in {docs_path}")
            print("💡 Add a PDF file to test image processing.")
    else:
        print(f"❌ Directory {docs_path} not found.")
        print("💡 Create the directory and add a PDF file for testing.")
