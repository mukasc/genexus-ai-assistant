import os
import json
from typing import Dict, Any
from dotenv import load_dotenv

# 1. Definição de Caminhos
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. Carregamento de Ambiente (.env)
root_env_path = os.path.join(ROOT_DIR, ".env")
load_dotenv(root_env_path)
backend_env_path = os.path.join(ROOT_DIR, "backend", ".env")
load_dotenv(backend_env_path)

# 3. Configuração White Label (JSON)
CONFIG_FILE = os.path.join(ROOT_DIR, "backend", "app_config.json")

# Estrutura padrão para fallback se o arquivo falhar
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
        "model_name": "gemini-1.5-flash",
        "temperature": 0.1,
        "system_prompt": "You are a helpful assistant."
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

# Variável Global que segura o estado COMPLETO do JSON
GLOBAL_STATE = {
    "active_profile": "default",
    "profiles": { "default": DEFAULT_CONFIG }
}

# Variável Global que segura APENAS a configuração do perfil ATUAL (para compatibilidade)
APP_CONFIG = {}

def load_app_config():
    """Carrega o JSON do disco e popula as variáveis globais"""
    global GLOBAL_STATE, APP_CONFIG
    
    if not os.path.exists(CONFIG_FILE):
        print(f"⚠️ Arquivo {CONFIG_FILE} não encontrado. Usando Default.")
        APP_CONFIG.update(DEFAULT_CONFIG)
        return

    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Detecta se é o formato novo (com profiles) ou antigo
        if "profiles" in data:
            GLOBAL_STATE = data
            active_key = data.get("active_profile", "default")
            active_conf = data["profiles"].get(active_key, DEFAULT_CONFIG)
            
            # Atualiza o dicionário APP_CONFIG in-place
            APP_CONFIG.clear()
            APP_CONFIG.update(active_conf)
            print(f"✅ Perfil carregado: {active_key}")
        else:
            # Formato legado (fallback)
            APP_CONFIG.clear()
            APP_CONFIG.update(data)
            
    except Exception as e:
        print(f"⚠️ ERRO ao ler config: {e}")
        APP_CONFIG.clear()
        APP_CONFIG.update(DEFAULT_CONFIG)

def save_active_profile(profile_key: str):
    """Salva a escolha do perfil no disco para persistir no restart"""
    global GLOBAL_STATE
    
    if profile_key not in GLOBAL_STATE["profiles"]:
        raise ValueError("Perfil não existe")
        
    GLOBAL_STATE["active_profile"] = profile_key
    
    # Atualiza o APP_CONFIG em memória
    APP_CONFIG.clear()
    APP_CONFIG.update(GLOBAL_STATE["profiles"][profile_key])
    
    # Salva no disco
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(GLOBAL_STATE, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Erro ao salvar config: {e}")

# Carga inicial
load_app_config()

# 4. Variáveis Globais de Configuração
API_KEY = os.getenv("GEMINI_API_KEY")