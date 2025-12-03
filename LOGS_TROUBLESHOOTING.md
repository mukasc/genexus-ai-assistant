# 🔧 Troubleshooting - Visualizador de Logs

## ✅ Problema Resolvido!

O erro "404" foi corrigido. O visualizador de logs agora está funcionando corretamente.

---

## 🎯 Como Testar

### 1. Teste Direto da API

```bash
# Teste local
curl http://localhost:8001/api/logs?lines=5

# Teste público (use seu domínio)
curl https://code-reviewer-18.preview.emergentagent.com/api/logs?lines=5
```

**Resposta esperada:**
```json
{
  "logs": [...],
  "total": 5
}
```

### 2. Teste no Preview

1. Abra o preview do app
2. Clique no botão "📊 Ver Logs" no sidebar
3. Os logs devem aparecer

---

## ⚠️ Erros Comuns e Soluções

### Erro: "404 Not Found"

**Causa:** Endpoint não encontrado ou URL incorreta

**Solução:**
```bash
# 1. Verificar se backend está rodando
curl http://localhost:8001/api/health

# 2. Verificar endpoint de logs
curl http://localhost:8001/api/logs?lines=5

# 3. Se não funcionar, reiniciar backend
sudo supervisorctl restart backend
```

### Erro: "Erro ao buscar logs"

**Causa:** Conexão com backend falhou

**Verificações:**
```bash
# Status do backend
sudo supervisorctl status backend

# Logs de erro do backend
tail -20 /var/log/supervisor/backend.err.log

# Testar conectividade
curl -I http://localhost:8001/api/health
```

**Solução:**
```bash
# Reiniciar backend
sudo supervisorctl restart backend

# Aguardar 5 segundos
sleep 5

# Testar novamente
curl http://localhost:8001/api/logs?lines=5
```

### Erro: "Nenhum log encontrado"

**Causa:** Arquivo de log vazio ou filtros muito restritivos

**Solução:**
1. Remover todos os filtros
2. Selecionar "Todos" no level
3. Limpar busca
4. Aumentar número de linhas para 500
5. Clicar em "Atualizar"

**Verificação manual:**
```bash
# Ver se arquivo de log existe e tem conteúdo
ls -lh /var/log/supervisor/backend.out.log

# Ver últimas linhas
tail -10 /var/log/supervisor/backend.out.log
```

### Erro: CORS / Network Error

**Causa:** Frontend não consegue acessar backend

**Solução:**
```bash
# 1. Verificar CORS no backend
grep -A 5 "CORS" /app/backend/server.py

# 2. Verificar BACKEND_URL no frontend
cat /app/frontend/.env

# 3. Reiniciar ambos
sudo supervisorctl restart backend frontend
```

### Modal não abre

**Causa:** JavaScript error no frontend

**Verificação:**
1. Abrir console do navegador (F12)
2. Ir para aba "Console"
3. Procurar erros em vermelho

**Solução:**
```bash
# Verificar logs do frontend
tail -50 /var/log/supervisor/frontend.err.log

# Reiniciar frontend
sudo supervisorctl restart frontend
```

### Auto-refresh não funciona

**Causa:** Intervalo não configurado corretamente

**Solução:**
1. Desmarcar checkbox "Auto-refresh"
2. Aguardar 3 segundos
3. Marcar novamente
4. Logs devem atualizar automaticamente

### Logs desatualizados

**Causa:** Cache do navegador

**Solução:**
1. Forçar reload: Ctrl+F5 (ou Cmd+Shift+R no Mac)
2. Ou limpar cache do navegador
3. Fechar e reabrir modal de logs

---

## 🔍 Diagnóstico Completo

Se nada funcionar, execute este diagnóstico:

```bash
# Script de diagnóstico
cat > /tmp/diagnose-logs.sh << 'EOF'
#!/bin/bash

echo "=== Diagnóstico do Visualizador de Logs ==="
echo ""

echo "1. Status dos serviços:"
sudo supervisorctl status backend frontend
echo ""

echo "2. Backend está respondendo?"
curl -s -o /dev/null -w "HTTP Status: %{http_code}\n" http://localhost:8001/api/health
echo ""

echo "3. Endpoint de logs funciona?"
curl -s -o /dev/null -w "HTTP Status: %{http_code}\n" http://localhost:8001/api/logs?lines=1
echo ""

echo "4. Arquivo de log existe?"
ls -lh /var/log/supervisor/backend.out.log
echo ""

echo "5. Últimas 3 linhas do log:"
tail -3 /var/log/supervisor/backend.out.log
echo ""

echo "6. BACKEND_URL no frontend:"
cat /app/frontend/.env | grep BACKEND
echo ""

echo "7. Teste completo da API:"
curl -s http://localhost:8001/api/logs?lines=2 | python3 -m json.tool
echo ""

echo "=== Fim do diagnóstico ==="
EOF

chmod +x /tmp/diagnose-logs.sh
/tmp/diagnose-logs.sh
```

**Interpretação dos resultados:**

✅ **Tudo OK se:**
- Ambos serviços em RUNNING
- HTTP Status: 200 para ambos endpoints
- Arquivo de log existe com tamanho > 0
- API retorna JSON válido

❌ **Há problema se:**
- Algum serviço em STOPPED/FATAL
- HTTP Status: 404, 500, ou erro de conexão
- Arquivo de log não existe
- API retorna erro

---

## 🛠️ Soluções Rápidas

### Reset Completo

```bash
# Reiniciar tudo
sudo supervisorctl restart all

# Aguardar serviços iniciarem
sleep 10

# Testar
curl http://localhost:8001/api/logs?lines=5
```

### Limpar Logs Antigos

```bash
# Se logs estiverem muito grandes
sudo truncate -s 0 /var/log/supervisor/backend.out.log

# Reiniciar para gerar novos logs
sudo supervisorctl restart backend

# Aguardar
sleep 5

# Gerar alguns logs
curl http://localhost:8001/api/health
curl http://localhost:8001/api/health
curl http://localhost:8001/api/health
```

### Verificar Permissões

```bash
# Backend precisa ler os logs
ls -la /var/log/supervisor/backend.out.log

# Se permissões estiverem erradas
sudo chmod 644 /var/log/supervisor/backend.out.log
```

---

## 📞 Suporte

Se nenhuma solução funcionar:

1. **Coletar informações:**
```bash
/tmp/diagnose-logs.sh > /tmp/diagnostic-report.txt
cat /tmp/diagnostic-report.txt
```

2. **Verificar logs de erro:**
```bash
tail -100 /var/log/supervisor/backend.err.log
tail -100 /var/log/supervisor/frontend.err.log
```

3. **Testar manualmente:**
```bash
# Ver se Python está OK
cd /app/backend
python3 -c "import logging; from pythonjsonlogger import jsonlogger; print('OK')"

# Ver se endpoint está registrado
grep -n "@app.get.*logs" /app/backend/server.py
```

---

## ✅ Checklist de Verificação

Antes de reportar problema:

- [ ] Backend em RUNNING
- [ ] Frontend em RUNNING
- [ ] `curl http://localhost:8001/api/health` retorna 200
- [ ] `curl http://localhost:8001/api/logs?lines=5` retorna JSON
- [ ] Arquivo `/var/log/supervisor/backend.out.log` existe
- [ ] Console do navegador não mostra erros
- [ ] Tentei Ctrl+F5 no navegador
- [ ] Tentei fechar e reabrir modal de logs

---

## 🎯 Teste Final

Depois de aplicar qualquer solução:

```bash
# 1. Reiniciar tudo
sudo supervisorctl restart all && sleep 10

# 2. Testar API
curl -s http://localhost:8001/api/logs?lines=5 | python3 -m json.tool

# 3. Se retornar JSON válido = SUCESSO! ✅
```

Agora abra o preview e teste o botão "📊 Ver Logs"!

---

**Problema persiste? Execute o diagnóstico completo e compartilhe o resultado.**
