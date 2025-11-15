import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma


# Carrega a API Key do arquivo .env
load_dotenv("keys.env")

def run_ingestion():
    # 1. Carregar Documentos
    print("Carregando documentos...")
    docs_path = "./docs"
    documents = []
    
    # Percorre todos os PDFs na pasta 'docs'
    for filename in os.listdir(docs_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(docs_path, filename))
            documents.extend(loader.load())
            
    if not documents:
        print("Nenhum PDF encontrado na pasta 'docs'. Abortando.")
        return

    # 2. Segmentação (Chunking) - Ajuste o chunk_size e chunk_overlap conforme a necessidade
    print(f"Segmentando {len(documents)} páginas em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200, 
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total de chunks criados: {len(chunks)}")

    # 3. Criação de Embeddings e Indexação
    print("Criando embeddings com o GoogleGenerativeAI e indexando no ChromaDB...")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("A chave GEMINI_API_KEY não foi carregada. Verifique seu arquivo keys.env.")
    
    # Modelo robusto para criação de vetores de texto
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=api_key
        ) 
    
    # Cria o Vector Store e armazena os vetores localmente
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db" 
    )
    vectorstore.persist()
    print("Ingestão concluída. Banco de dados vetorial salvo em ./chroma_db")

if __name__ == "__main__":
    run_ingestion()
