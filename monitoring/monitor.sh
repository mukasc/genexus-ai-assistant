#!/bin/bash

echo "========================================"
echo "  🔴 Monitoramento em Tempo Real"
echo "========================================"
echo "Pressione Ctrl+C para sair"
echo ""

if command -v jq &> /dev/null; then
    tail -f /var/log/supervisor/backend.out.log | while read line; do
        # Colorir por level
        level=$(echo "$line" | jq -r '.level' 2>/dev/null)
        
        if [ "$level" = "error" ]; then
            echo "$line" | jq -C '.'
        elif [ "$level" = "warning" ]; then
            echo "$line" | jq -C '.'
        else
            echo "$line" | jq -C '.' 2>/dev/null || echo "$line"
        fi
    done
else
    echo "⚠️  jq não instalado. Mostrando logs sem formatação."
    echo "Para melhor visualização: sudo apt-get install -y jq"
    echo ""
    tail -f /var/log/supervisor/backend.out.log
fi
