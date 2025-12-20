import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import APP_CONFIG, API_KEY, ROOT_DIR
from app.logging_config import logger
from app.core.embeddings import OptimizedEmbeddings

# Variáveis Globais de Estado
rag_chain = None
vectorstore_instance = None 

def get_optimized_embeddings():
    if not API_KEY: raise ValueError("GEMINI_API_KEY missing")
    base = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=API_KEY, 
        transport="rest", 
        task_type="retrieval_document"
    )
    return OptimizedEmbeddings(base, use_cache=True, batch_size=10, delay=2.0)

def get_vectorstore():
    # 1. Tenta pegar do .env primeiro (Prioridade Máxima)
    env_path = os.getenv('CHROMA_DB_PATH')
    
    # 2. Se não tiver no .env, tenta do JSON, se não, usa default
    json_path = APP_CONFIG.get('storage', {}).get('persist_directory', 'data/chroma_db')
    
    # Define o diretório final
    p_dir = env_path if env_path else json_path
    
    # Define o nome da coleção
    coll_name = APP_CONFIG.get('storage', {}).get('collection_name', 'default_collection')

    # Garante caminho absoluto
    if not os.path.isabs(p_dir):
        abs_dir = os.path.join(ROOT_DIR, p_dir)
    else:
        abs_dir = p_dir
    
    logger.info(f"Carregando ChromaDB em: {abs_dir}")
    
    return Chroma(
        collection_name=coll_name, 
        persist_directory=abs_dir, 
        embedding_function=get_optimized_embeddings()
    )

def initialize_rag_system():
    global rag_chain, vectorstore_instance
    if not API_KEY: return {"success": False, "error": "No API Key"}

    try:
        vectorstore_instance = get_vectorstore()
        retriever = vectorstore_instance.as_retriever(search_kwargs={"k": 3})
        
        # Configs do JSON
        model_name = APP_CONFIG.get('llm', {}).get('model_name', 'gemini-2.5-flash')
        temp = APP_CONFIG.get('llm', {}).get('temperature', 0.1)
        sys_prompt = APP_CONFIG.get('llm', {}).get('system_prompt', "Context: {context} Question: {question}")
        
        llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=temp, 
            google_api_key=API_KEY, 
            max_retries=1, 
            transport="rest"
        )
        
        prompt = ChatPromptTemplate.from_template(sys_prompt)
        
        def format_docs(docs):
            content = "\n\n".join(d.page_content for d in docs)
            if docs:
                logger.info(f"Retrieved {len(docs)} docs, total chars: {len(content)}", extra={"docs_found": True})
            else:
                logger.warning("No docs found for query", extra={"docs_found": False})
            return content if docs else "No context."
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return {"success": True, "message": f"Initialized '{APP_CONFIG.get('identity', {}).get('app_name')}'"}
    except Exception as e:
        logger.error(f"RAG Init Error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=2, min=5, max=30), 
    retry=retry_if_exception_type(Exception)
)
def run_chain_with_retry(chain, message):
    return chain.invoke(message)