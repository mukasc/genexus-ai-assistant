# 📊 Visualizador de Logs Web

## ✨ Acesso aos Logs via Navegador

Agora você pode visualizar os logs diretamente no seu navegador, sem precisar de terminal ou Docker!

---

## 🚀 Como Acessar

### No Preview do Aplicativo:

1. **Abra o preview do seu app** (o frontend React)

2. **Localize o botão no sidebar**:
   ```
   📊 Ver Logs
   ```

3. **Clique no botão** - Uma janela modal aparecerá com os logs!

---

## 🎨 Recursos do Visualizador

### 1️⃣ **Filtros Inteligentes**

**Por Número de Linhas:**
- 50, 100, 200, ou 500 linhas

**Por Level:**
- Todos
- Errors apenas
- Warnings apenas  
- Info apenas
- Debug apenas

**Busca de Texto:**
- Digite qualquer termo para filtrar

### 2️⃣ **Auto-Refresh**

- Ative o checkbox "Auto-refresh (3s)"
- Os logs atualizam automaticamente a cada 3 segundos
- Perfeito para monitorar em tempo real

### 3️⃣ **Interface Visual**

**Cores por Level:**
- 🔴 **Errors**: Vermelho
- 🟠 **Warnings**: Laranja
- 🔵 **Info**: Azul
- 🟣 **Debug**: Roxo

**Informações Exibidas:**
- ✅ Level do log
- ✅ Nome do logger
- ✅ Função que gerou o log
- ✅ Linha do código
- ✅ Mensagem completa
- ✅ Campos extras (contexto)

### 4️⃣ **Ações Disponíveis**

**🔄 Atualizar:**
- Busca logs mais recentes manualmente

**💾 Download:**
- Baixa todos os logs filtrados em arquivo `.log`

**📋 Copiar:**
- Cada log tem um botão para copiar (formato JSON)

---

## 💡 Casos de Uso

### 1. Debugar Erro Específico

```
1. Abrir visualizador de logs
2. Selecionar "Errors" no filtro de level
3. Ver últimos erros com contexto completo
4. Copiar log para análise detalhada
```

### 2. Monitorar Chat em Tempo Real

```
1. Abrir visualizador
2. Digitar "chat" na busca
3. Ativar auto-refresh
4. Usar o chat e ver logs aparecendo
```

### 3. Verificar Ingestion

```
1. Abrir visualizador
2. Buscar por "ingest"
3. Ver logs de upload/processamento
4. Verificar chunks criados
```

### 4. Análise de Performance

```
1. Abrir visualizador
2. Selecionar 500 linhas
3. Buscar por função específica
4. Ver contexto (response_length, etc)
```

---

## 📸 O Que Você Verá

### Cabeçalho
```
📊 Application Logs                    [×]
```

### Controles
```
Linhas: [100 ▼]  Level: [Todos ▼]  🔍 [Buscar...]  
☐ Auto-refresh (3s)  [🔄 Atualizar]  [💾 Download]
```

### Estatísticas
```
Total: 100  Errors: 2  Warnings: 5
```

### Exemplo de Log
```
┌──────────────────────────────────────────┐
│ 📝 INFO  server  health_check:210  [📋] │
│ Health check requested                    │
│ 📦 server                                 │
│ status: "healthy"  api_configured: true  │
└──────────────────────────────────────────┘
```

---

## 🎯 Vantagens sobre Terminal

| Característica | Terminal | Visualizador Web |
|----------------|----------|------------------|
| **Acesso** | SSH/Terminal | Navegador |
| **Instalação** | jq, scripts | Nenhuma |
| **Interface** | Texto puro | Visual colorida |
| **Filtros** | grep, jq | Clicks |
| **Auto-refresh** | Manual | Automático |
| **Compartilhar** | Copiar texto | URL do preview |
| **Download** | Redirecionamento | 1 click |
| **Mobile** | Difícil | Responsivo |

---

## 🔧 Recursos Técnicos

### Endpoint da API
```
GET /api/logs?lines=100&level=error&search=chat
```

**Parâmetros:**
- `lines`: número de linhas (padrão: 100)
- `level`: filtro por level (opcional)
- `search`: busca de texto (opcional)

**Resposta:**
```json
{
  "logs": [...],
  "total": 100
}
```

### Campos do Log JSON
```json
{
  "timestamp": null,
  "level": "info",
  "name": "server",
  "message": "Chat request received",
  "module": "server",
  "funcName": "chat",
  "lineno": 220,
  "message_length": 45
}
```

---

## 📱 Responsivo

O visualizador funciona em:
- ✅ Desktop (experiência completa)
- ✅ Tablet (ajustado)
- ✅ Mobile (layout vertical)

---

## 🆘 Troubleshooting

### "Nenhum log encontrado"

**Causas possíveis:**
- Filtros muito restritivos
- Backend acabou de reiniciar
- Arquivo de log vazio

**Solução:**
1. Remover filtros (Level: Todos, Busca: vazia)
2. Aumentar número de linhas
3. Clicar em "Atualizar"

### "Erro ao buscar logs"

**Causas:**
- Backend não está rodando
- Problema de conexão

**Solução:**
```bash
# Verificar backend
curl http://localhost:8001/api/health

# Reiniciar se necessário
sudo supervisorctl restart backend
```

### Logs não atualizam automaticamente

**Solução:**
1. Desativar e reativar "Auto-refresh"
2. Verificar se backend está respondendo
3. Recarregar página do preview

---

## 💡 Dicas Profissionais

### 1. Durante Desenvolvimento
```
- Abra visualizador em aba separada
- Ative auto-refresh
- Filtre por função que está testando
- Monitore erros em tempo real
```

### 2. Para Debugging
```
- Filtre por level="error"
- Copie log completo
- Analise campos extras
- Busque por stack trace
```

### 3. Para Demos
```
- Mostre logs durante apresentação
- Auto-refresh para efeito live
- Demonstre rastreabilidade
- Mostre campos contextuais
```

### 4. Para Análise
```
- Download logs
- Analise offline
- Compare períodos diferentes
- Procure padrões
```

---

## 🎉 Comparação Completa

### Antes (Terminal)
```bash
ssh servidor
tail -f /var/log/supervisor/backend.out.log | jq '.'
# Difícil compartilhar
# Sem filtros visuais
# Requer acesso SSH
```

### Agora (Web)
```
1. Abrir preview
2. Clicar "📊 Ver Logs"
3. Ver, filtrar, baixar
# Visual, fácil, acessível
```

---

## 🔗 Links Relacionados

- **Logs via Terminal**: `/app/monitoring/ACESSO_LOGS.md`
- **Stack PLG**: `/app/monitoring/README.md`
- **Guia de Erros**: `/app/ERROR_HANDLING_GUIDE.md`

---

## ✨ Features Futuras

Planejadas para próximas versões:
- [ ] Gráficos de logs por tempo
- [ ] Alertas de erro em tempo real
- [ ] Export para CSV/Excel
- [ ] Highlight de sintaxe JSON
- [ ] Busca com regex
- [ ] Filtros salvos
- [ ] Dashboard de métricas

---

**Acesse agora!** 🚀

Abra seu preview e clique em **📊 Ver Logs** no sidebar!
