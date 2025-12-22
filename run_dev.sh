#!/bin/bash

# GeneXus AI Assistant - Auto Dev Starter
# Inicia diretamente o servidor backend sem menus

# --- CONFIGURAÇÃO GIT AUTOMÁTICA ---
git config --system user.name "Murillo Petry"
git config --system user.email "mukasc@gmail.com"

# 1. Configurar Ambiente se necessário
if [ ! -f "backend/.env" ]; then
    if [ -f "backend/.env.example" ]; then
        cp backend/.env.example backend/.env
        echo "📝 .env criado."
    fi
fi

# 2. Ativar Virtual Env e Instalar deps
if [ ! -d "venv" ]; then
    echo "📦 Criando venv..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -q -r backend/requirements.txt
else
    source venv/bin/activate
fi

# 3. Executar o Servidor (Hot Reload)
echo "🚀 Iniciando Backend na porta 8001..."
cd backend
uvicorn main:app --host 0.0.0.0 --port 8001 --reload