import os
import json
from typing import Dict, Any
from dotenv import load_dotenv

# 1. Definição de Caminhos
# backend/app/config.py -> backend/app -> backend -> Raiz do Projeto
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. Carregamento de Ambiente (.env) - Ordem de Prioridade
# A. Tenta carregar o .env da RAÍZ do projeto (Onde provavelmente estão suas chaves reais)
root_env_path = os.path.join(ROOT_DIR, ".env")
load_dotenv(root_env_path)

# B. Tenta carregar o .env da pasta BACKEND (Sobrescreve/Complementa se existir)
backend_env_path = os.path.join(ROOT_DIR, "backend", ".env")
load_dotenv(backend_env_path)

# Debug: Verificação no console ao iniciar
print(f"--- CONFIG DEBUG ---")
print(f"Raiz do Projeto: {ROOT_DIR}")
print(f"Lendo .env da Raiz? {'SIM' if os.path.exists(root_env_path) else 'NÃO'}")
print(f"Lendo .env do Backend? {'SIM' if os.path.exists(backend_env_path) else 'NÃO'}")
# ------------------

# 3. Configuração White Label (JSON)
CONFIG_FILE = os.path.join(ROOT_DIR, "backend", "app_config.json")

DEFAULT_CONFIG = {
    "identity": {
        "app_name": "AI Assistant (Default)",
        "app_subtitle": "System running in default mode",
        "welcome_message": "Config file not found.",
        "primary_color": "#333333",
        "secondary_color": "#555555",
        "logo_emoji": "⚠️"
    },
    "llm": {
        "model_name": "gemini-2.5-flash",
        "temperature": 0.1,
        "system_prompt": "You are a helpful assistant. Context: {context} Question: {question}"
    },
    "storage": {
        "collection_name": "default_collection",
        "persist_directory": "data/chroma_db"
    },
    "ingestion": {
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
}

def load_app_config() -> Dict[str, Any]:
    """Carrega configuração do JSON ou usa Default se falhar"""
    if not os.path.exists(CONFIG_FILE):
        print(f"⚠️ AVISO: Arquivo {CONFIG_FILE} não encontrado. Usando Default.")
        return DEFAULT_CONFIG
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ ERRO: Falha ao ler JSON ({e}). Usando Default.")
        return DEFAULT_CONFIG

APP_CONFIG = load_app_config()

# 4. Variáveis Globais de Configuração
API_KEY = os.getenv("GEMINI_API_KEY")

# Verificação Final
if not API_KEY:
    print("❌ ERRO CRÍTICO: GEMINI_API_KEY não encontrada nos arquivos .env!")
else:
    print("✅ GEMINI_API_KEY carregada com sucesso.")