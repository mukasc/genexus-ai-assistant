#!/bin/bash

echo "========================================"
echo "  ⚠️  Análise de Erros"
echo "========================================"
echo ""

LOG_FILE="/var/log/supervisor/backend.out.log"
LINES=1000

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Arquivo de log não encontrado: $LOG_FILE"
    exit 1
fi

if command -v jq &> /dev/null; then
    echo "📊 Analisando últimas $LINES linhas..."
    echo ""
    
    # Total de erros
    ERROR_COUNT=$(tail -$LINES "$LOG_FILE" | jq -r 'select(.level=="error")' 2>/dev/null | wc -l)
    echo "Total de erros: $ERROR_COUNT"
    
    echo ""
    echo "=== Erros por Tipo ==="
    tail -$LINES "$LOG_FILE" | jq -r 'select(.level=="error") | .error_type // "general"' 2>/dev/null | sort | uniq -c | sort -rn
    
    echo ""
    echo "=== Erros por Função ==="
    tail -$LINES "$LOG_FILE" | jq -r 'select(.level=="error") | .funcName' 2>/dev/null | sort | uniq -c | sort -rn | head -5
    
    echo ""
    echo "=== Últimos 5 Erros ==="
    tail -$LINES "$LOG_FILE" | jq -C 'select(.level=="error") | {timestamp, funcName, message}' 2>/dev/null | tail -5
    
    echo ""
    echo "=== Rate Limits ==="
    RATE_LIMIT_COUNT=$(tail -$LINES "$LOG_FILE" | jq -r 'select(.error_type=="rate_limit")' 2>/dev/null | wc -l)
    echo "Total: $RATE_LIMIT_COUNT"
    
else
    echo "⚠️  jq não instalado. Análise básica:"
    echo ""
    ERROR_COUNT=$(tail -$LINES "$LOG_FILE" | grep -c '"level":"error"' || echo "0")
    echo "Total de erros: $ERROR_COUNT"
    echo ""
    echo "Últimos 5 erros:"
    tail -$LINES "$LOG_FILE" | grep '"level":"error"' | tail -5
    echo ""
    echo "💡 Instale jq para análise detalhada: sudo apt-get install -y jq"
fi

echo ""
echo "========================================"
