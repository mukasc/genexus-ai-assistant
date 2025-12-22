#!/bin/bash

# Script de Automação de Testes (Pytest)
# Garante que o ambiente está pronto e executa a bateria de testes.

echo "======================================================="
echo "🧪 GeneXus AI Assistant - Testes Automatizados"
echo "======================================================="
echo ""

# 1. Verificar e Ativar Virtual Environment
if [ -d "venv" ]; then
    echo "🔄 Ativando ambiente virtual..."
    source venv/bin/activate
else
    echo "⚠️  Aviso: Pasta 'venv' não encontrada na raiz."
    echo "   Tentando utilizar o Python do sistema..."
fi

# 2. Verificar Instalação do Pytest
if ! command -v pytest &> /dev/null; then
    echo "📥 Pytest não encontrado. Instalando dependências de teste..."
    pip install -q -r backend/requirements.txt
fi

# 3. Configurar PYTHONPATH
# Garante que o Python encontra o módulo 'app' dentro de 'backend/'
export PYTHONPATH=$PYTHONPATH:$(pwd)/backend

# 4. Executar os Testes
echo "🚀 Iniciando execução..."
echo ""

# Executa o pytest:
# -v: Verbose (mostra cada teste individualmente)
# --disable-warnings: Limpa a saída removendo avisos de depreciação de libs
pytest -v --disable-warnings

# Captura o código de saída do pytest (0 = Sucesso, 1 = Falha)
EXIT_CODE=$?

echo ""
echo "======================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCESSO: Todos os testes passaram!"
else
    echo "❌ FALHA: Alguns testes não passaram. Verifique o log acima."
fi
echo "======================================================="

exit $EXIT_CODE