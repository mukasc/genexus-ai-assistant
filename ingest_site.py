import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
import re 
import requests

# Dependências do Selenium
import time 
from selenium import webdriver 
from selenium.webdriver.chrome.service import Service 
from selenium.webdriver.common.by import By 


# --- 1. SETUP DE AMBIENTE E API KEY ---
load_dotenv("keys.env") 
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("ERRO: A variável de ambiente GEMINI_API_KEY não está configurada.")
    exit()

# Define o limite de artigos a serem indexados
MAX_ARTICLES_TO_INDEX = 50 
# Define quantas páginas de busca serão rastreadas
MAX_PAGES_TO_SCAN = 10 

# URLs (NOVA ESTRATÉGIA DE BUSCA PAGINADA)
URL_SEARCH_BASE = "https://docs.genexus.com/en/hsearch?+category%3AGeneXus+18+Help"
BASE_DOCS_URL = "https://docs.genexus.com"
# >> AJUSTE ESTE CAMINHO PARA ONDE SEU chromedriver.exe ESTÁ SALVO <<
CHROME_DRIVER_PATH = r"D:\genexus-ai-assistant\chromedriver.exe" 


def get_driver_with_selenium():
    """Inicializa e configura o driver do Chrome."""
    
    options = webdriver.ChromeOptions()
    
    # Configurações para emular um navegador real
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    
    service = Service(CHROME_DRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=options)
    
    # Espera implícita para elementos aparecerem (útil para JS)
    driver.implicitly_wait(5)
    
    return driver
    
        
def run_ingestion():
    # Lista para armazenar novos documentos Web
    documents = []
    article_links = set()
    driver = None 

    # --- 2. CARREGAMENTO DA DOCUMENTAÇÃO WEB (SELENIUM e Paginação) ---
    print(f"\nIniciando Scraper nas páginas de Busca do GeneXus 18 (até {MAX_PAGES_TO_SCAN} páginas)...")
    
    current_page = 1
    
    try:
        # Inicializa o driver fora do loop para reutilizá-lo
        driver = get_driver_with_selenium()

        while current_page <= MAX_PAGES_TO_SCAN and len(article_links) < MAX_ARTICLES_TO_INDEX:
            
            # 1. Constrói a URL Paginada
            if current_page == 1:
                page_url = URL_SEARCH_BASE
            else:
                page_url = f"{URL_SEARCH_BASE},{current_page}"
                
            print(f"\n -> Processando Página {current_page}: {page_url}")

            # 2. Navega e espera o conteúdo carregar
            driver.get(page_url)
            # Espera fixa para que os resultados da busca dinâmica sejam injetados
            time.sleep(10) 
            
            # 3. Extração de Links Diretos (>> EXTRAÇÃO COM SELETOR ESPECÍFICO <<)
            
            # CSS Selector: Procura por tags <a> que são filhas diretas de 
            # <span> com a classe "Search__Title" (que tem o padrão TITLE_XXXX)
            CSS_SELECTOR = 'span.Search__Title > a'
            
            links_elements = driver.find_elements(By.CSS_SELECTOR, CSS_SELECTOR)
                
            if not links_elements:
                if current_page > 1:
                    print(" -> Nenhuma ligação encontrada. Fim da paginação.")
                else:
                    print(" -> Nenhuma ligação encontrada na primeira página.")
                break 

            print(f" -> {len(links_elements)} links de artigos encontrados na página {current_page}.")

            new_links_on_page = 0
            for link_element in links_elements:
                relative_url = link_element.get_attribute('href')
                
                # Filtro para garantir que é um artigo wiki e não um arquivo ou âncora
                if (relative_url and 
                    '/en/wiki?' in relative_url and 
                    '#' not in relative_url and 
                    not any(ext in relative_url for ext in ['.png', '.jpg', '.gif', '.css', '.js', '.svg'])
                ):
                    full_url = relative_url
                    
                    # Garante que a URL seja absoluta
                    if full_url.startswith('/'):
                        full_url = BASE_DOCS_URL + full_url

                    # Adiciona link se não for duplicata
                    if full_url not in article_links:
                        article_links.add(full_url)
                        new_links_on_page += 1
                        
                        if len(article_links) >= MAX_ARTICLES_TO_INDEX:
                            break
            
            print(f" -> {new_links_on_page} novos links adicionados. Total: {len(article_links)}/{MAX_ARTICLES_TO_INDEX}")

            if len(article_links) >= MAX_ARTICLES_TO_INDEX:
                break
                
            # Prepara para a próxima iteração
            current_page += 1

    except Exception as e:
        print(f"ERRO durante a navegação ou extração: {e}. Interrompendo.")
    finally:
        # 4. Fechar o driver
        if driver:
            driver.quit() 
        
    # --- Continuação da Ingestão ---
    
    if not article_links:
        print("\nNenhum link de artigo foi extraído. Finalizando ingestão.")
        return

    # 3. Carregar o CONTEÚDO COMPLETO de cada artigo selecionado
    # ... (o restante do script permanece o mesmo: loading, chunking, embeddings)
    
    article_documents = []
    print("\nIniciando carregamento do conteúdo dos artigos selecionados...")
    
    for i, link in enumerate(article_links):
        print(f" -> Indexando Artigo {i+1}/{len(article_links)}: {link}")
        
        # WebBaseLoader é usado para carregar o conteúdo do artigo (URL por URL)
        article_loader = WebBaseLoader(link)
        # Tenta carregar, ignorando falhas individuais de URL (timeout, etc)
        try:
            article_documents.extend(article_loader.load())
        except Exception as e:
            print(f" !! ERRO ao carregar {link}: {e}")
            
    documents.extend(article_documents)
    print(f"Total de artigos completos carregados: {len(article_documents)}")


    if not documents:
        print("\nNenhum documento Web foi carregado. Finalizando ingestão.")
        return

    # --- 3. DIVISÃO (CHUNKING) ---
    print(f"\nDividindo {len(documents)} novos documentos em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    new_chunks = text_splitter.split_documents(documents)
    print(f"Total de novos Chunks criados: {len(new_chunks)}")

    # --- 4. EMBEDDING E ARMAZENAMENTO (CHROMA DB) ---
    print("\nInicializando Embeddings e manipulando o Chroma DB...")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=API_KEY
    )

    try:
        # Tenta carregar a base de dados existente (para combinar)
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        print("Base de dados existente (incluindo PDFs) carregada com sucesso.")
        
        # Adiciona os novos chunks da Web à base existente
        print("Adicionando os novos documentos Web à base existente...")
        vectorstore.add_documents(new_chunks)
        
    except Exception as e:
        # Se a base não existir, cria uma nova
        print(f"Erro ao carregar base existente ({e}). Criando nova base do zero...")
        vectorstore = Chroma.from_documents(
            documents=new_chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
    # Salva as alterações, persistindo tanto os dados antigos quanto os novos
    vectorstore.persist()
    print("\n✅ Ingestão concluída com sucesso!")
    print(f"Os dados (antigos e novos) estão agora combinados em: ./chroma_db")

if __name__ == "__main__":
    run_ingestion()