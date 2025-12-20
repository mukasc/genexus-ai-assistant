import os
from operator import itemgetter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Imports para Memória (NOVO)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import APP_CONFIG, API_KEY, ROOT_DIR
from app.logging_config import logger
from app.core.embeddings import OptimizedEmbeddings

# Variáveis Globais de Estado
rag_chain = None
vectorstore_instance = None 

# --- MEMÓRIA EM RAM (NOVO) ---
# Dicionário para guardar histórico: { "session_id": ChatMessageHistory() }
# Em produção real, isso poderia ser substituído por Redis
session_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]
# -----------------------------

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
    """
    Retorna a instância do ChromaDB (Local).
    """
    embeddings = get_optimized_embeddings()
    
    # 1. Definição do Caminho
    env_path = os.getenv('CHROMA_DB_PATH')
    json_path = APP_CONFIG.get('storage', {}).get('persist_directory', 'data/chroma_db')
    
    # Prioriza .env > json > default
    p_dir = env_path if env_path else json_path
    
    # Garante caminho absoluto a partir da raiz
    if not os.path.isabs(p_dir):
        abs_dir = os.path.join(ROOT_DIR, p_dir)
    else:
        abs_dir = p_dir
        
    # 2. Nome da Coleção
    coll_name = APP_CONFIG.get('storage', {}).get('collection_name', 'default_collection')
    
    logger.info(f"Carregando ChromaDB em: {abs_dir}")
    
    return Chroma(
        collection_name=coll_name, 
        persist_directory=abs_dir, 
        embedding_function=embeddings
    )

def initialize_rag_system():
    global rag_chain, vectorstore_instance
    if not API_KEY: return {"success": False, "error": "No API Key"}

    try:
        vectorstore_instance = get_vectorstore()
        
        # --- CONFIGURAÇÃO DE RETRIEVAL ---
        retrieval_conf = APP_CONFIG.get('retrieval', {})
        k_docs = retrieval_conf.get('k', 4)
        score_thresh = retrieval_conf.get('score_threshold', 0.8) # Mantendo o filtro alto

        logger.info(f"Configurando Retriever: k={k_docs}, threshold={score_thresh}")

        # Usa 'similarity_score_threshold' para filtrar lixo
        retriever = vectorstore_instance.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": score_thresh,
                "k": k_docs
            }
        )
        
        # Configs do JSON
        model_name = APP_CONFIG.get('llm', {}).get('model_name', 'gemini-1.5-flash')
        temp = APP_CONFIG.get('llm', {}).get('temperature', 0.1)
        # Pega o System Prompt do JSON
        sys_instructions = APP_CONFIG.get('llm', {}).get('system_prompt', "You are a helpful assistant.")
        
        llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=temp, 
            google_api_key=API_KEY, 
            max_retries=1, 
            transport="rest"
        )
        
        # --- NOVO PROMPT TEMPLATE COM HISTÓRICO ---
        # Substituímos from_template por from_messages para injetar o histórico corretamente
        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_instructions),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
        
        # Função auxiliar para formatar texto
        def format_docs_text(docs):
            if not docs:
                # Se o filtro remover tudo, loga um aviso
                logger.warning("Nenhum documento atingiu o score mínimo de relevância.", extra={"docs_found": False})
                return "" # Retorna vazio, o Prompt deve lidar com isso ("I didn't find...")

            content = "\n\n".join(d.page_content for d in docs)
            logger.info(f"Retrieved {len(docs)} relevant docs", extra={"docs_found": True})
            return content

        # Configuração da Chain Interna (Antes da Memória)
        # Processa: {question, chat_history} -> {response, sources}
        chain_with_docs = (
            RunnableParallel({
                # Recupera docs usando apenas a pergunta atual
                "docs": itemgetter("question") | retriever,
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history")
            })
            .assign(context=lambda x: format_docs_text(x["docs"]))
            | {
                "response": prompt | llm | StrOutputParser(),
                "sources": itemgetter("docs") # Preserva os objetos Document
            }
        )

        # --- APLICAÇÃO DA MEMÓRIA ---
        # Envolve a chain básica com o gerenciador de histórico
        rag_chain = RunnableWithMessageHistory(
            chain_with_docs,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
            output_messages_key="response"
        )
        
        return {"success": True, "message": f"Initialized '{APP_CONFIG.get('identity', {}).get('app_name')}' with Memory & Sources"}
    except Exception as e:
        logger.error(f"RAG Init Error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# Função de execução atualizada para aceitar config (onde vai o session_id)
@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=2, min=5, max=30), 
    retry=retry_if_exception_type(Exception)
)
def run_chain_with_retry(chain, input_data, config):
    return chain.invoke(input_data, config=config)