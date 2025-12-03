#!/bin/bash

echo "========================================"
echo "  🚀 Quick Start - Logs"
echo "========================================"
echo ""

# Verificar jq
if ! command -v jq &> /dev/null; then
    echo "📦 Instalando jq..."
    sudo apt-get update -qq
    sudo apt-get install -y jq
    echo "✅ jq instalado!"
    echo ""
fi

# Tornar scripts executáveis
chmod +x /app/monitoring/*.sh

echo "✅ Scripts configurados!"
echo ""
echo "Comandos disponíveis:"
echo ""
echo "1. Ver logs formatados:"
echo "   /app/monitoring/view-logs.sh"
echo ""
echo "2. Monitorar em tempo real:"
echo "   /app/monitoring/monitor.sh"
echo ""
echo "3. Analisar erros:"
echo "   /app/monitoring/analyze-errors.sh"
echo ""
echo "4. Ver log direto:"
echo "   tail -f /var/log/supervisor/backend.out.log | jq '.'"
echo ""
echo "========================================"
echo "💡 Execute qualquer comando acima!"
echo "========================================"
