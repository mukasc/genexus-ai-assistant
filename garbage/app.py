import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load API Key from .env file
load_dotenv()

# Configuration from environment variables with defaults
API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))

# --- Initial Configuration ---
st.set_page_config(page_title="GeneXus AI Assistant (RAG)", layout="wide")
st.title("🤖 GeneXus AI Assistant (RAG Prototype)")
st.caption("GeneXus specialist powered by official documentation and Gemini API.")

# Optimized Prompt Template
PROMPT_TEMPLATE = """
You are the **GeneXus Code Assistant**, a senior GeneXus expert. Your mission is to provide complete and robust solutions, following best practices.

**CODE AND RESPONSE GUIDELINES:**
1.  **GeneXus Priority:** Always generate code **EXCLUSIVELY in GeneXus syntax**. Use code blocks (```genexus).
2.  **Focus on Structured Data:** Prioritize information found in **tables, property lists, and syntax definitions** within the 'CONTEXT'. This textual data is your source of truth, compensating for the absence of visual diagrams.
3.  **Contextual Inference:** If the 'CONTEXT' describes a process or data flow (which may have originally been a diagram), **infer the logical flow** and translate it to the correct GeneXus syntax (e.g., *parameters, Procedure commands*).
4.  **Strict Fidelity to Context (RAG):** Your response must be **entirely based on the provided 'CONTEXT'**.
5.  **Intelligent Rejection:** If the context is insufficient, decline to answer.
6.  **Language: Must interpret all languages but the response must always be in PT-BR or the language provided.

CONTEXT (GeneXus Documentation and Tutorials):
{context}

USER QUESTION: {question}
"""


@st.cache_resource
def get_retriever():
    """Load the vector database and create the Retriever."""
    # Ensure API Key is available
    if not API_KEY:
        st.error("⚠️ The GEMINI_API_KEY environment variable is not configured. Please check your .env file.")
        st.info("💡 Copy .env.example to .env and add your Gemini API key.")
        st.stop()
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=API_KEY
        )
    except Exception as e:
        st.error(f"❌ Error initializing embeddings: {e}")
        st.stop()
    
    # Connect to persisted Vector Store
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
        # k: retrieves the k most relevant chunks
        return vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    except Exception as e:
        st.error(f"❌ Error loading the database. Run 'python ingest.py' first.")
        st.error(f"Error details: {e}")
        st.info("💡 Make sure you have run the ingestion script to create the vector database.")
        st.stop()


def format_docs(docs):
    """Format retrieved documents into a simple string."""
    if not docs:
        return "No relevant context found."
    return "\n\n".join(doc.page_content for doc in docs)


# Initialize application
try:
    # 1. Get the Retriever
    retriever = get_retriever()
    
    # 2. Configure the LLM (Gemini)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.1,
        google_api_key=API_KEY
    )
    
    # 3. Create the RAG Chain (LangChain Expression Language - LCEL)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # The RAG pipe: Context -> Prompt -> LLM -> Response
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
except Exception as e:
    st.error(f"❌ Error initializing the application: {e}")
    st.stop()

# --- Streamlit Interface ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture new user input
if prompt_input := st.chat_input("Ask something about GeneXus..."):
    # Validate input
    if not prompt_input.strip():
        st.warning("⚠️ Please enter a valid question.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking like a GeneXus specialist..."):
                try:
                    response = rag_chain.invoke(prompt_input)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar for additional information
with st.sidebar:
    st.header("📊 Prototype Status")
    st.markdown(f"**RAG Framework:** LangChain")
    st.markdown(f"**LLM:** Gemini 2.0 Flash")
    st.markdown(f"**Vector Store:** ChromaDB")
    st.markdown(f"**Retrieval:** Top {RETRIEVAL_K} chunks")
    
    st.divider()
    
    st.header("ℹ️ Information")
    st.markdown("""
    This assistant uses:
    - **RAG (Retrieval Augmented Generation)** to search GeneXus documentation
    - **Gemini AI** to generate specialized responses
    - **ChromaDB** to store and search document vectors
    """)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
