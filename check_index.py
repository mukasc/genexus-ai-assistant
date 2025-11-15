import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. CONFIGURAÇÃO DE AMBIENTE ---
# Garante que a chave da API do Gemini seja carregada
load_dotenv("keys.env") 
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("ERRO: GEMINI_API_KEY não configurada. Não é possível carregar o DB.")
    exit()

# --- 2. INICIALIZAÇÃO E CARREGAMENTO DO ÍNDICE ---
# Inicializa embeddings (precisa ser o mesmo modelo usado para criar o índice)
embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=API_KEY
)

try:
    print("Carregando ChromaDB da pasta ./chroma_db...")
    # 1. Carrega o Vector Store persistido
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    # 2. Busca de similaridade (query simples)
    # Buscamos um termo relevante para o novo conteúdo (GeneXus 18)
    results = vectorstore.similarity_search("GeneXus 18 Super Apps", k=10) 
    
    print(f"\n✅ Total de documentos (chunks) no índice: {vectorstore._collection.count()}")
    print("\n🔗 10 Exemplos de Artigos Indexados (Verificando a URL de Origem):")
    print("-" * 60)
    
    # 3. Itera sobre os resultados para mostrar a fonte (source)
    for i, doc in enumerate(results):
        source_url = doc.metadata.get('source', 'Fonte não encontrada')
        
        # Classifica a origem
        if "docs.genexus.com" in source_url:
            source_type = "🌐 WEB ARTICLE"
        elif source_url.endswith(".pdf"):
            source_type = "📄 LOCAL PDF"
        else:
            source_type = "❓ OUTRA FONTE"

        # Exibe a origem e a URL
        print(f"[{source_type}]")
        print(f"   {source_url}")
        print(f"   Conteúdo Inicial: \"{doc.page_content[:80]}...\"")
        print("-" * 60)
            
except Exception as e:
    print(f"\nFalha ao carregar o ChromaDB: {e}")