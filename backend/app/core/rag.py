import os
import time
from operator import itemgetter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Imports para Memória
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError

from app.config import APP_CONFIG, API_KEY, ROOT_DIR
from app.logging_config import logger
from app.core.embeddings import OptimizedEmbeddings

# Variáveis Globais de Estado
rag_chain = None
vectorstore_instance = None 

# --- CONTROLE DE ROTAÇÃO DE MODELOS ---
current_model_index = 0

def get_current_model_name():
    """Retorna o nome do modelo atual baseado no índice de rotação."""
    llm_conf = APP_CONFIG.get('llm', {})
    fallback_list = llm_conf.get('fallback_order', [])
    
    # Se não tiver lista, usa o único definido
    if not fallback_list:
        return llm_conf.get('model_name', 'gemini-1.5-flash')
    
    # Garante que o índice está dentro dos limites
    safe_index = current_model_index % len(fallback_list)
    return fallback_list[safe_index]

def switch_to_next_model():
    """Avança para o próximo modelo da lista e reinicia o RAG."""
    global current_model_index, rag_chain
    
    llm_conf = APP_CONFIG.get('llm', {})
    fallback_list = llm_conf.get('fallback_order', [])
    
    if not fallback_list or len(fallback_list) <= 1:
        logger.warning("Tentativa de Fallback, mas não há lista de modelos alternativos configurada.")
        return False

    # Avança o índice
    old_model = fallback_list[current_model_index % len(fallback_list)]
    current_model_index = (current_model_index + 1) % len(fallback_list)
    new_model = fallback_list[current_model_index]
    
    logger.warning(f"⚠️ ROTAÇÃO DE MODELO: {old_model} falhou (Cota/Erro). Alternando para -> {new_model}")
    
    # Força reinicialização
    rag_chain = None
    initialize_rag_system()
    return True
# --------------------------------------

# --- MEMÓRIA EM ARQUIVO ---
def get_session_history(session_id: str) -> BaseChatMessageHistory:
       
                                                       
                                             
       
                                     
    sessions_dir = os.path.join(ROOT_DIR, "data", "sessions")
    
                                
    if not os.path.exists(sessions_dir):
            
        try: os.makedirs(sessions_dir)
        except: pass
                                                                 
    
                                                               
    file_path = os.path.join(sessions_dir, f"{session_id}.json")
    
                                                                       
    return FileChatMessageHistory(file_path)

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
       
                                             
       
    embeddings = get_optimized_embeddings()
    
                               
    env_path = os.getenv('CHROMA_DB_PATH')
    json_path = APP_CONFIG.get('storage', {}).get('persist_directory', 'data/chroma_db')
    
                                    
    p_dir = env_path if env_path else json_path
    
                                               
    if not os.path.isabs(p_dir):
        abs_dir = os.path.join(ROOT_DIR, p_dir)
    else:
        abs_dir = p_dir
        
                          
    coll_name = APP_CONFIG.get('storage', {}).get('collection_name', 'default_collection')
    
    logger.info(f"Carregando ChromaDB em: {abs_dir}")
    return Chroma(collection_name=coll_name, persist_directory=abs_dir, embedding_function=embeddings)
                  
                                   
                                   
                                     
     

def initialize_rag_system():
    global rag_chain, vectorstore_instance
    if not API_KEY: return {"success": False, "error": "No API Key"}

    try:
        vectorstore_instance = get_vectorstore()
        
        # Retrieval Config
        retrieval_conf = APP_CONFIG.get('retrieval', {})
        k_docs = retrieval_conf.get('k', 4)
        score_thresh = retrieval_conf.get('score_threshold', 0.6)
        
        # Expansion Config
        expansion_conf = retrieval_conf.get('query_expansion', {})
        use_expansion = expansion_conf.get('enabled', False)
        expansion_count = expansion_conf.get('count', 3)

        logger.info(f"Retriever: k={k_docs}, threshold={score_thresh}")

                                                            
        base_retriever = vectorstore_instance.as_retriever(
            search_type="similarity_score_threshold",
                           
            search_kwargs={"score_threshold": score_thresh, "k": k_docs}
                           
             
        )
        
        # --- LLM SELECTION (DINÂMICO) ---
        model_name = get_current_model_name()
        temp = APP_CONFIG.get('llm', {}).get('temperature', 0.1)
                                      
        sys_instructions = APP_CONFIG.get('llm', {}).get('system_prompt', "You are a helpful assistant.")
        
        logger.info(f"Inicializando LLM com modelo ATIVO: {model_name}")

        llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=temp, 
            google_api_key=API_KEY, 
            max_retries=1, # Deixa o retry externo lidar com a rotação
            transport="rest"
        )

        # Retrieval Logic (Expansion or Standard)
        retriever_chain = None
        
        if use_expansion:
            try:
                                                             
                                                                   
                expansion_prompt = ChatPromptTemplate.from_template(
                    "Generate {count} different versions of the user question to retrieve relevant documents.\nOriginal: {question}"
                                                                               
                                                                        
                                                   
                 
                
                                    
                                     
                          
                                        
                                               
                )
                generate_queries = (expansion_prompt | llm | StrOutputParser() | (lambda x: x.split("\n")))

                                                                                
                def expanded_retrieval(input_dict):
                    question = input_dict["question"]
                    
                                      
                    queries = generate_queries.invoke({"question": question, "count": expansion_count})
                                                           
                    queries = [question] + [q.strip() for q in queries if q.strip()]
                    
                                                            
                    
                                                            
                    all_docs = []
                    for q in queries:
                        all_docs.extend(base_retriever.invoke(q))
                                             
                    
                                                                         
                    unique_docs = []
                    seen = set()
                    for doc in all_docs:
                        if doc.page_content not in seen:
                            seen.add(doc.page_content)
                            unique_docs.append(doc)
                    
                    return unique_docs[:k_docs*2]

                retriever_chain = RunnableLambda(expanded_retrieval)
            except Exception as e:
                logger.error(f"Expansion failed: {e}")
                retriever_chain = None

                                                                             
        if not retriever_chain:
            retriever_chain = itemgetter("question") | base_retriever

        # Prompt
                                                                                              
        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_instructions),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
        
                                               
        def format_docs_text(docs):
            if not docs:
                                                         
                logger.warning("Nenhum documento atingiu o score mínimo.")
                return "" 

            content = "\n\n".join(d.page_content for d in docs)
                                                                                           
            return content

                                                             
                                                                   
        chain_with_docs = (
            RunnableParallel({
                                                              
                "docs": retriever_chain,
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history")
            })
            .assign(context=lambda x: format_docs_text(x["docs"]))
            | {
                "response": prompt | llm | StrOutputParser(),
                "sources": itemgetter("docs")
            }
        )

                                         
                                                                 
        rag_chain = RunnableWithMessageHistory(
            chain_with_docs,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
            output_messages_key="response"
        )
        
                                                                                                   
        return {"success": True, "message": f"Initialized with {model_name}"}
    except Exception as e:
        logger.error(f"RAG Init Error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# --- EXECUTOR COM LÓGICA DE FALLBACK ---
async def run_chain_with_fallback(input_data, config, is_streaming=False):
    """
    Tenta executar a chain. Se der erro 429, troca de modelo e tenta de novo.
    Tenta no máximo 3 modelos diferentes antes de desistir.
    """
    max_model_switches = 3
    
    for attempt in range(max_model_switches):
        try:
            if not rag_chain:
                initialize_rag_system()
                
            if is_streaming:
                # Para streaming, retornamos o gerador diretamente
                # Se falhar durante o stream, o catch abaixo pega
                return rag_chain.astream(input_data, config=config)
            else:
                # Execução normal
                return await rag_chain.ainvoke(input_data, config=config)
                
        except Exception as e:
            error_msg = str(e)
            # Verifica se é erro de cota (429) ou Resource Exhausted
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                logger.error(f"🚨 Cota excedida no modelo atual. Tentativa {attempt+1}/{max_model_switches}")
                if switch_to_next_model():
                    time.sleep(1) # Pequena pausa para respirar
                    continue # Tenta de novo com o novo modelo
            
            # Se não for erro de cota, ou se não tiver mais modelos, explode o erro
            raise e