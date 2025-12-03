# 📊 Como Acessar os Logs de Monitoramento

## 🎯 Opções Disponíveis

### Opção 1: Ver Logs Diretamente (SEM Docker) ✅ Funciona Agora

Os logs JSON estruturados já estão sendo gerados. Você pode visualizá-los diretamente:

#### Ver Logs em Tempo Real

```bash
# Logs do backend (JSON estruturado)
tail -f /var/log/supervisor/backend.out.log

# Com formatação colorida (se tiver jq instalado)
tail -f /var/log/supervisor/backend.out.log | jq '.'

# Apenas erros
tail -f /var/log/supervisor/backend.out.log | jq 'select(.level=="error")'

# Logs do frontend
tail -f /var/log/supervisor/frontend.out.log

# Todos os logs do supervisor
tail -f /var/log/supervisor/*.log
```

#### Ver Últimas N Linhas

```bash
# Últimas 50 linhas do backend
tail -50 /var/log/supervisor/backend.out.log

# Últimas 100 linhas formatadas
tail -100 /var/log/supervisor/backend.out.log | jq '.'
```

#### Filtrar Logs Específicos

```bash
# Buscar por "chat"
grep -i "chat" /var/log/supervisor/backend.out.log

# Buscar erros
grep -i "error" /var/log/supervisor/backend.out.log

# Buscar rate limit
grep -i "rate_limit" /var/log/supervisor/backend.out.log

# Buscar ingestion
grep -i "ingest" /var/log/supervisor/backend.out.log
```

#### Análise de Logs JSON com jq

```bash
# Instalar jq (se não tiver)
sudo apt-get install -y jq

# Filtrar apenas campo message
tail -100 /var/log/supervisor/backend.out.log | jq -r '.message'

# Ver apenas erros com timestamp
tail -100 /var/log/supervisor/backend.out.log | jq 'select(.level=="error") | {timestamp, message, error}'

# Contar logs por level
tail -100 /var/log/supervisor/backend.out.log | jq -r '.level' | sort | uniq -c

# Ver logs de ingestion
tail -100 /var/log/supervisor/backend.out.log | jq 'select(.funcName | contains("ingest"))'
```

---

### Opção 2: Stack PLG Completa (COM Docker) 🐳 Requer Instalação

Para usar Grafana + Loki + Promtail, você precisa instalar Docker primeiro.

#### Passo 1: Instalar Docker

```bash
# Seguir o guia completo
cat /app/monitoring/DOCKER_SETUP.md

# OU instalação rápida (Ubuntu/Debian):
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

#### Passo 2: Iniciar a Stack

```bash
cd /app/monitoring
docker compose up -d
```

#### Passo 3: Verificar Status

```bash
docker compose ps

# Deve mostrar:
# loki        running    3100/tcp
# promtail    running    
# grafana     running    0.0.0.0:3001->3000/tcp
```

#### Passo 4: Acessar Grafana

1. **Abrir navegador**: http://localhost:3001
2. **Login**: 
   - Usuário: `admin`
   - Senha: `admin`
3. **Primeira vez**: Trocar senha ou skip
4. **Ir para "Explore"** (ícone de bússola no menu lateral)

#### Passo 5: Fazer Queries no Loki

Na página Explore do Grafana:

**Query 1: Todos os logs do backend**
```logql
{job="backend"}
```

**Query 2: Apenas erros**
```logql
{job="backend"} | json | level="error"
```

**Query 3: Chat requests**
```logql
{job="backend"} |~ "Chat request"
```

**Query 4: Rate limits**
```logql
{job="backend"} | json | error_type="rate_limit"
```

**Query 5: Ingestion events**
```logql
{job="backend"} |~ "ingestion"
```

**Query 6: Últimos 5 minutos**
```logql
{job="backend"} [5m]
```

---

### Opção 3: Scripts Auxiliares (Criados para Você) 🛠️

Criei alguns scripts para facilitar:

#### Script 1: Ver Logs Formatados

```bash
# Criar script
cat > /app/monitoring/view-logs.sh << 'EOF'
#!/bin/bash

echo "=== Logs do Backend (últimas 20 linhas) ==="
tail -20 /var/log/supervisor/backend.out.log | jq '.' 2>/dev/null || tail -20 /var/log/supervisor/backend.out.log

echo ""
echo "=== Erros Recentes ==="
grep -i error /var/log/supervisor/backend.out.log | tail -5 | jq '.' 2>/dev/null || grep -i error /var/log/supervisor/backend.out.log | tail -5
EOF

chmod +x /app/monitoring/view-logs.sh
/app/monitoring/view-logs.sh
```

#### Script 2: Monitorar em Tempo Real

```bash
# Criar script
cat > /app/monitoring/monitor.sh << 'EOF'
#!/bin/bash

echo "Monitorando logs do backend..."
echo "Pressione Ctrl+C para sair"
echo ""

tail -f /var/log/supervisor/backend.out.log | while read line; do
    echo "$line" | jq '.' 2>/dev/null || echo "$line"
done
EOF

chmod +x /app/monitoring/monitor.sh
/app/monitoring/monitor.sh
```

#### Script 3: Análise de Erros

```bash
# Criar script
cat > /app/monitoring/analyze-errors.sh << 'EOF'
#!/bin/bash

echo "=== Análise de Erros ==="
echo ""

echo "Total de erros nas últimas 1000 linhas:"
tail -1000 /var/log/supervisor/backend.out.log | grep -c '"level":"error"' || echo "0"

echo ""
echo "Tipos de erros:"
tail -1000 /var/log/supervisor/backend.out.log | jq -r 'select(.level=="error") | .error_type' 2>/dev/null | sort | uniq -c || echo "Nenhum erro encontrado"

echo ""
echo "Últimos 5 erros:"
tail -1000 /var/log/supervisor/backend.out.log | jq 'select(.level=="error") | {timestamp, funcName, message}' 2>/dev/null | tail -5
EOF

chmod +x /app/monitoring/analyze-errors.sh
/app/monitoring/analyze-errors.sh
```

---

## 🎨 Comparação das Opções

| Característica | Logs Diretos | Stack PLG |
|----------------|--------------|-----------|
| **Requer instalação** | ❌ Não | ✅ Sim (Docker) |
| **Disponível agora** | ✅ Sim | ⏳ Após instalar |
| **Visualização** | Terminal | Interface Web |
| **Filtros** | grep, jq | LogQL queries |
| **Dashboard** | ❌ | ✅ Grafana |
| **Histórico** | Arquivos log | Loki database |
| **Tempo real** | tail -f | Grafana Live |
| **Melhor para** | Debug rápido | Análise profunda |

---

## 📋 Comandos Úteis de Acesso

### Acesso Rápido (Copie e Cole)

```bash
# Ver logs formatados
tail -50 /var/log/supervisor/backend.out.log | jq '.'

# Monitorar erros em tempo real
tail -f /var/log/supervisor/backend.out.log | jq 'select(.level=="error")'

# Ver últimas requisições de chat
tail -100 /var/log/supervisor/backend.out.log | jq 'select(.funcName=="chat")'

# Estatísticas de log levels
tail -500 /var/log/supervisor/backend.out.log | jq -r '.level' | sort | uniq -c

# Ver ingestion events
tail -200 /var/log/supervisor/backend.out.log | jq 'select(.funcName | contains("ingest"))'
```

---

## 🚀 Recomendação

### Para Começar Agora (Sem instalar nada):

1. **Instale jq** para melhor visualização:
   ```bash
   sudo apt-get install -y jq
   ```

2. **Use o script de monitoramento**:
   ```bash
   tail -f /var/log/supervisor/backend.out.log | jq '.'
   ```

3. **Teste a aplicação** e veja os logs aparecerem!

### Para Análise Profissional (Instalar depois):

1. **Instale Docker** (5-10 minutos)
2. **Inicie a stack PLG** (1 minuto)
3. **Acesse Grafana** no navegador
4. **Crie dashboards personalizados**

---

## 💡 Dicas Práticas

### 1. Ver Logs Durante Testes

Em um terminal:
```bash
tail -f /var/log/supervisor/backend.out.log | jq '.'
```

Em outro terminal:
```bash
# Fazer requisição
curl http://localhost:8001/api/health

# Upload de PDF
curl -X POST http://localhost:8001/api/ingest-pdf -F "files=@teste.pdf"
```

### 2. Debugar Erros Específicos

```bash
# Ver último erro com contexto completo
tail -1000 /var/log/supervisor/backend.out.log | jq 'select(.level=="error")' | tail -1 | jq '.'
```

### 3. Verificar Performance

```bash
# Ver todas as requisições de chat
grep "Chat request" /var/log/supervisor/backend.out.log | wc -l

# Ver tempo médio (se adicionar timing logs)
grep "response_time" /var/log/supervisor/backend.out.log | jq -r '.response_time'
```

---

## 🆘 Troubleshooting

### "jq: command not found"

```bash
sudo apt-get update
sudo apt-get install -y jq
```

### "Permission denied" ao acessar logs

```bash
# Adicionar permissões de leitura
sudo chmod +r /var/log/supervisor/*.log

# OU usar sudo
sudo tail -f /var/log/supervisor/backend.out.log
```

### Logs muito grandes

```bash
# Limpar logs antigos
sudo truncate -s 0 /var/log/supervisor/backend.out.log

# OU rotacionar logs
sudo logrotate -f /etc/logrotate.conf
```

### Docker não funciona

```bash
# Verificar instalação
docker --version

# Se não estiver instalado
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

---

## 📚 Próximos Passos

1. ✅ **Agora**: Use logs diretos com jq
2. 🐳 **Depois**: Instale Docker para Grafana
3. 📊 **Avançado**: Crie dashboards customizados
4. 🔔 **Futuro**: Configure alertas no Grafana

---

**Comece agora mesmo!**
```bash
# Ver logs em tempo real
tail -f /var/log/supervisor/backend.out.log | jq '.'
```
