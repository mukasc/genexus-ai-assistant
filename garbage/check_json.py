import json
import os

CONFIG_FILE = "app_config.json"

if not os.path.exists(CONFIG_FILE):
    print(f"❌ ERRO: O arquivo '{CONFIG_FILE}' não existe nesta pasta!")
    exit(1)

try:
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("✅ JSON Válido (Sintaxe correta).")
    
    # Valida campos obrigatórios
    required_keys = ["identity", "llm", "storage"]
    missing = [k for k in required_keys if k not in data]
    
    if missing:
        print(f"❌ ERRO LÓGICO: Faltam seções obrigatórias: {missing}")
    else:
        print(f"✅ Estrutura Lógica parece ok. App Name: {data['identity'].get('app_name')}")

except json.JSONDecodeError as e:
    print(f"❌ ERRO DE SINTAXE NO JSON:\n{e}")
except Exception as e:
    print(f"❌ ERRO DESCONHECIDO: {e}")