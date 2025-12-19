# Arquivo: check_models.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Erro: API Key não configurada.")
    exit(1)

genai.configure(api_key=api_key)

print(f"Consultando modelos (API Key: {api_key[:5]}...)...")
print("-" * 30)

try:
    for m in genai.list_models():
        # Versão compatível com bibliotecas antigas e novas
        name = m.name
        methods = m.supported_generation_methods
        
        if 'generateContent' in methods:
            print(f"MODELO DISPONÍVEL: {name}")
            
except Exception as e:
    print(f"Erro fatal: {e}")