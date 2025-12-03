#!/bin/bash

echo "========================================"
echo "  📊 Logs do GeneXus AI Assistant"
echo "========================================"
echo ""

if command -v jq &> /dev/null; then
    echo "=== Últimas 20 linhas (Backend) ==="
    tail -20 /var/log/supervisor/backend.out.log | jq -C '.' 2>/dev/null
    
    echo ""
    echo "=== Erros Recentes ==="
    tail -100 /var/log/supervisor/backend.out.log | jq -C 'select(.level=="error")' 2>/dev/null | tail -5
    
    echo ""
    echo "=== Estatísticas por Level ==="
    echo "INFO: $(tail -200 /var/log/supervisor/backend.out.log | jq -r '.level' 2>/dev/null | grep -c 'info')"
    echo "ERROR: $(tail -200 /var/log/supervisor/backend.out.log | jq -r '.level' 2>/dev/null | grep -c 'error')"
    echo "WARNING: $(tail -200 /var/log/supervisor/backend.out.log | jq -r '.level' 2>/dev/null | grep -c 'warning')"
else
    echo "⚠️  jq não está instalado. Instalando..."
    echo "Execute: sudo apt-get install -y jq"
    echo ""
    echo "=== Logs sem formatação ==="
    tail -20 /var/log/supervisor/backend.out.log
fi

echo ""
echo "========================================"
echo "💡 Dica: Use './monitor.sh' para tempo real"
echo "========================================"
