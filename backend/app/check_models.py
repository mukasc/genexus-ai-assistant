import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# --- 1. Carregamento Robusto do .env ---
# Descobre onde este script está rodando
script_dir = Path(__file__).resolve().parent
backend_dir = script_dir.parent
root_dir = backend_dir.parent

print(f"📂 Procurando .env em:")
print(f"  - {script_dir}")
print(f"  - {backend_dir}")
print(f"  - {root_dir}")

# Tenta carregar de todos os locais possíveis
load_dotenv(script_dir / ".env")
load_dotenv(backend_dir / ".env")
load_dotenv(root_dir / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("\n❌ ERRO CRÍTICO: Variável GEMINI_API_KEY não encontrada.")
    print("   Certifique-se de que o arquivo .env existe em uma das pastas acima e contém a chave.")
    sys.exit(1)

print(f"\n✅ Chave encontrada: {api_key[:5]}...{api_key[-5:]}")

# --- 2. Consulta à API ---
try:
    genai.configure(api_key=api_key)
    
    print("\n📡 Consultando modelos disponíveis na API do Google...")
    print("(Isso pode levar alguns segundos...)\n")

    models = list(genai.list_models())
    found_flash = False
    
    print(f"{'NOME DO MODELO (ID)':<40} | {'NOME DE EXIBIÇÃO'}")
    print("-" * 70)
    
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            print(f"{m.name:<40} | {m.display_name}")
            if "flash" in m.name.lower():
                found_flash = True

    print("-" * 70)

    if found_flash:
        print("\n✅ SUGESTÃO: Atualize seu 'backend/app_config.json' com um destes:")
        print('   "model_name": "models/gemini-1.5-flash"')
        print('   "model_name": "models/gemini-1.5-flash-latest"')
        print('   "model_name": "models/gemini-1.5-flash-001"')
    else:
        print("\n⚠️  Nenhum modelo 'Flash' detectado. Use 'models/gemini-pro'.")

except Exception as e:
    print(f"\n❌ Erro de conexão com o Google: {e}")
    if "403" in str(e):
        print("   -> Verifique se a API Key é válida e se a API 'Generative Language' está ativada no Google Cloud Console.")