import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Carrega a API Key do arquivo .env
load_dotenv("keys.env")

API_KEY = os.getenv("GEMINI_API_KEY")

# --- Configurações Iniciais ---
st.set_page_config(page_title="GeneXus AI Assistant (RAG)", layout="wide")
st.title("🤖 GeneXus AI Assistant (Protótipo RAG)")
st.caption("Especialista em GeneXus alimentado pela documentação oficial e Gemini API.")

# O Prompt Template é a otimização crucial para especializar o LLM
PROMPT_TEMPLATE_OLD = """
Você é um assistente de programação **perito em GeneXus**. Sua função é auxiliar o desenvolvedor a escrever código, modelar objetos e entender conceitos GeneXus.
**Seu foco deve ser na sintaxe e nos objetos GeneXus (Transactions, Data Providers, Procedures, Web Panels) e não em linguagens de programação subjacentes (Java, C#, etc.).**

**INSTRUÇÕES:**
1.  Use **APENAS** as informações contidas no 'CONTEXTO' abaixo para formular sua resposta.
2.  Responda de forma clara e técnica.
3.  Quando gerar código GeneXus, use blocos de código (` ``` `) e especifique o tipo (ex: ` ```genexus` ou ` ```sql`).
4.  Se o contexto não for suficiente, diga educadamente que, com a sua base de conhecimento atual, você não pode responder à pergunta específica sobre GeneXus.
5.  Mantenha a resposta focada no tema GeneXus.

CONTEXTO:
{context}

PERGUNTA DO USUÁRIO: {question}
"""

PROMPT_TEMPLATE_OTIMIZED = """
Você é o **GeneXus Code Assistant**, um especialista sênior em GeneXus (todas as versões) e engenharia de software Low-Code.
Sua missão é fornecer soluções completas, robustas e que sigam as **melhores práticas de modelagem e programação GeneXus**.

**DIRETRIZES DE CÓDIGO E RESPOSTA:**
1.  **Prioridade GeneXus:** Sempre que a pergunta for sobre implementação ou sintaxe, priorize a criação de código **EXCLUSIVAMENTE em sintaxe GeneXus**.
2.  **Formato:** O código GeneXus deve ser envolto em blocos de código (` ```genexus`) para clareza. Para regras SQL/Data Selectors, use (` ```sql`).
3.  **Melhores Práticas:** Se o contexto recuperado mencionar otimizações (ex: uso de For Each com condições *inferred*, minimização de acessos a banco de dados), **integre-as** na sua sugestão de código.
4.  **Estrita Fidelidade ao Contexto (RAG):** Sua resposta deve ser **inteiramente baseada no 'CONTEXTO'** fornecido. Não invente ou combine informações de conhecimento geral.
5.  **Rejeição Inteligente:** Se o contexto for insuficiente ou irrelevante, recuse-se a responder, informando que a base de conhecimento (documentação) não cobre o tópico.
6.  **Foco em Objeto:** Para requisições de modelagem (ex: 'criar um Data Provider'), entregue o código completo da estrutura do objeto.

CONTEXTO (Documentação GeneXus e Tutoriais):
{context}

PERGUNTA DO USUÁRIO: {question}
"""

PROMPT_TEMPLATE = """
Você é o **GeneXus Code Assistant**, um especialista sênior em GeneXus. Sua missão é fornecer soluções completas e robustas, seguindo as melhores práticas.

**DIRETRIZES DE CÓDIGO E RESPOSTA:**
1.  **Prioridade GeneXus:** Sempre gere código **EXCLUSIVAMENTE em sintaxe GeneXus**. Use blocos de código (` ```genexus`).
2.  **Foco em Dados Estruturados:** Priorize informações encontradas em **tabelas, listas de propriedades e definições de sintaxe** dentro do 'CONTEXTO'. Estes dados textuais são a sua fonte de verdade, compensando a ausência de diagramas visuais.
3.  **Inferência Contextual:** Se o 'CONTEXTO' descrever um processo ou fluxo de dados (que pode ter sido originalmente um diagrama), **infira o fluxo lógico** e traduza-o para a sintaxe GeneXus correta (ex: *parâmetros, comandos de Procedure*).
4.  **Estrita Fidelidade ao Contexto (RAG):** Sua resposta deve ser **inteiramente baseada no 'CONTEXTO'** fornecido.
5.  **Rejeição Inteligente:** Se o contexto for insuficiente, recuse-se a responder.
6.  **Idioma: Deve interpretar todos os idiomas que conhece mas a resposta deve ser sempre em PT-BR ou no idioma fornecido.

CONTEXTO (Documentação GeneXus e Tutoriais):
{context}

PERGUNTA DO USUÁRIO: {question}
"""


@st.cache_resource
def get_retriever():
    """Carrega o banco de dados vetorial e cria o Retriever."""
    # Garante que a API Key esteja disponível
    if not API_KEY:
        st.error("A variável de ambiente GEMINI_API_KEY não está configurada.")
        st.stop()
        
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=API_KEY
        )
    
    # Conecta ao Vector Store persistido
    try:
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        # k=3: busca os 3 chunks mais relevantes
        return vectorstore.as_retriever(search_kwargs={"k": 3}) 
    except Exception as e:
        st.error(f"Erro ao carregar o banco de dados. Execute 'python ingest.py'. Erro: {e}")
        st.stop()

# 1. Obter o Retriever
retriever = get_retriever()

# 2. Configurar o LLM (Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

# 3. Criar a Cadeia RAG (LangChain Expression Language - LCEL)
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

def format_docs(docs):
    """Formata os documentos recuperados em uma string simples."""
    return "\n\n".join(doc.page_content for doc in docs)

# O pipe RAG: Contexto -> Prompt -> LLM -> Resposta
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Interface Streamlit ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibir histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capturar nova entrada do usuário
if prompt_input := st.chat_input("Pergunte algo sobre GeneXus..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Gerar resposta da IA
    with st.chat_message("assistant"):
        with st.spinner("Pensando como um especialista GeneXus..."):
            response = rag_chain.invoke(prompt_input)
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar para informações adicionais
st.sidebar.header("Status do Protótipo")
st.sidebar.markdown(f"**Framework RAG:** LangChain")
st.sidebar.markdown(f"**LLM:** Gemini 2.5 Flash")
st.sidebar.markdown(f"**Vector Store:** ChromaDB")
