# Arquivo: test_simple.py
import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Carrega variáveis
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

print("-" * 50)
print(f"Testando conexão com API Key: {api_key[:5]}...{api_key[-5:] if api_key else 'None'}")

if not api_key:
    print("ERRO: API Key não encontrada no .env")
    exit(1)

try:
    print("Iniciando chat simples com gemini-2.0-flash...")
    
    # MUDANÇA AQUI: Trocamos o modelo experimental pelo estável
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        google_api_key=api_key,
        temperature=0
    )
    
    # Chamada simples
    response = llm.invoke("Responda apenas com a palavra: CONECTADO")
    
    print("\nSUCESSO! Resposta do Google:")
    print(f">>> {response.content}")
    print("-" * 50)

except Exception as e:
    print(f"\nFALHA NA CONEXÃO:")
    print(e)