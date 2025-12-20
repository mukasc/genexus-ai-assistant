#!/bin/bash

# GeneXus AI Assistant - Enterprise Launcher
# Adapta-se à nova estrutura de pastas Modular (Backend em main.py)

echo "======================================================================"
echo "  🚀 GeneXus AI Assistant - Enterprise Boot"
echo "======================================================================"
echo ""

# 1. Verificar Ambiente (Backend)
if [ ! -f "backend/.env" ]; then
    echo "⚠️  backend/.env não encontrado."
    if [ -f "backend/.env.example" ]; then
        echo "📝 Criando .env a partir do exemplo..."
        cp backend/.env.example backend/.env
        echo "✅ Arquivo criado em backend/.env (Lembre-se de editar a API Key!)"
    fi
fi

# 2. Setup Virtual Environment
if [ ! -d "venv" ]; then
    echo "📦 Criando Virtual Environment na raiz..."
    python3 -m venv venv
fi

echo "🔄 Ativando venv..."
source venv/bin/activate

# 3. Instalar Dependências (Apontando para a pasta backend)
echo "📥 Instalando dependências do Backend..."
pip install -q -r backend/requirements.txt

# 4. Menu de Opções Atualizado
echo ""
echo "======================================================================"
echo "  O que você deseja fazer?"
echo "======================================================================"
echo "  1. Rodar Servidor API (Modo DEV - Auto Reload)"
echo "  2. Subir Infraestrutura Completa (Docker: API + Logs + Banco)"
echo "  3. Parar/Limpar Infraestrutura Docker"
echo "  4. Sair"
echo ""

read -p "Escolha (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Iniciando Servidor em Modo Desenvolvimento..."
        echo "🔥 Hot Reload ATIVADO (O servidor reinicia ao salvar arquivos)"
        echo "📡 API: http://localhost:8001"
        
        cd backend
        # AQUI ESTÁ O TRUQUE:
        # Em vez de 'python main.py', usamos 'uvicorn' direto.
        # main:app significa "arquivo main.py, objeto app"
        # --reload ativa o monitoramento de arquivos
        uvicorn main:app --host 0.0.0.0 --port 8001 --reload
        ;;
    2)
        echo ""
        echo "🐳 Subindo Containers (Docker Compose)..."
        docker-compose up -d --build
        echo ""
        echo "✅ Infraestrutura iniciada!"
        echo "📊 Grafana: http://localhost:3001"
        echo "📡 API: http://localhost:8001"
        ;;
    3)
        echo ""
        echo "🛑 Parando Containers..."
        docker-compose down
        ;;
    4)
        echo "👋 Até logo!"
        exit 0
        ;;
    *)
        echo "❌ Opção inválida"
        exit 1
        ;;
esac