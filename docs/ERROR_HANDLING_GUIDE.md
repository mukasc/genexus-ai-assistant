# 🛡️ Sistema de Tratamento de Erros - Guia Completo

## 📋 Visão Geral

O GeneXus AI Assistant agora possui um **sistema profissional de tratamento de erros** com notificações discretas e recuperação automática.

---

## ✨ Recursos Implementados

### 1. **Notificações Toast** 🍞
- Mensagens discretas que aparecem no canto inferior direito
- Não bloqueiam a interface
- Fecham automaticamente ou manualmente
- 4 tipos: Success, Error, Warning, Info

### 2. **Tratamento de Erro 429 (Rate Limit)** ⏱️
- Detecta limite de requisições da API
- Extrai tempo de retry automaticamente
- Desabilita botão temporariamente
- Contador regressivo visível
- Mensagem amigável em português

### 3. **Recuperação Automática** 🔄
- App não trava em caso de erro
- Estado de loading sempre removido
- Input preservado em erros de rate limit
- Usuário pode tentar novamente após o tempo

### 4. **Feedback Visual** 👁️
- Placeholder dinâmico: "Aguarde Xs..."
- Botão mostra countdown: "⏱️ 30s"
- Cores e ícones apropriados para cada situação

---

## 🎯 Como Funciona

### Erro 429 (Muitas Requisições)

**Antes:**
```
❌ Error: 429 You exceeded your current quota...
[Erro técnico aparece na tela]
[App pode travar]
```

**Agora:**
```
🔶 Toast aparece: "Muitas requisições no momento. Aguarde alguns segundos e tente novamente."
⏱️ Botão mostra: "⏱️ 30s"
📝 Input mostra: "Aguarde 30s..."
⏳ Countdown automático
✅ Botão reabilita automaticamente
```

### Outros Erros de API

**Erro de Autenticação:**
```
Toast: "Erro de autenticação com a API. Verifique suas credenciais."
Tipo: Error (vermelho)
```

**Erro de Conexão:**
```
Toast: "Erro de conexão. Verifique sua internet e tente novamente."
Tipo: Error (vermelho)
```

**Erro Genérico:**
```
Toast: "Erro ao processar a solicitação. Tente novamente."
Tipo: Error (vermelho)
```

### Sucessos

**Ingestão Bem-Sucedida:**
```
Toast: "Successfully ingested 2 PDF file(s). 45 fragmentos criados."
Tipo: Success (verde)
Duração: 5 segundos
```

---

## 🎨 Tipos de Toast

### Success ✅
```javascript
Cor: Verde (#10b981)
Ícone: ✅
Uso: Operações concluídas com sucesso
Duração: 5 segundos
```

### Error ⚠️
```javascript
Cor: Vermelho (#ef4444)
Ícone: ⚠️
Uso: Erros que não são rate limit
Duração: 5 segundos
```

### Warning ⏳
```javascript
Cor: Laranja (#f59e0b)
Ícone: ⏳
Uso: Rate limits, avisos temporários
Duração: Permanente (até manual close ou retry)
```

### Info ℹ️
```javascript
Cor: Azul (#3b82f6)
Ícone: ℹ️
Uso: Informações gerais
Duração: 5 segundos
```

---

## 🔧 Detalhes Técnicos

### Backend (server.py)

**ChatResponse atualizado:**
```python
class ChatResponse(BaseModel):
    response: str
    context_used: bool
    error: Optional[str] = None
    retry_after: Optional[int] = None  # Tempo em segundos
```

**Extração de retry_delay:**
```python
# Detecta erro 429
if "429" in error_str or "quota" in error_str.lower():
    # Extrai tempo de retry
    retry_match = re.search(r'retry in (\d+\.?\d*)', error_str)
    if retry_match:
        retry_after = int(float(retry_match.group(1)))
    else:
        retry_after = 30  # Default
    
    error_message = "Muitas requisições no momento. ..."
```

**Mensagens amigáveis:**
- ✅ Traduzidas para português
- ✅ Sem jargão técnico
- ✅ Ação clara para o usuário

### Frontend (App.js)

**Estados novos:**
```javascript
const [toast, setToast] = useState(null);
const [isRateLimited, setIsRateLimited] = useState(false);
const [retryTimer, setRetryTimer] = useState(0);
```

**Funções principais:**

1. **showToast()**
```javascript
showToast('Mensagem', 'success', 5000);
```

2. **startRetryTimer()**
```javascript
startRetryTimer(30); // 30 segundos
// Inicia countdown automático
// Desabilita botão
// Atualiza UI a cada segundo
```

3. **handleSubmit() melhorado**
```javascript
// Verifica rate limit
if (response.data.retry_after) {
    showToast(response.data.error, 'warning', 0);
    startRetryTimer(response.data.retry_after);
    // Preserva input do usuário
    setInput(userInput);
}
```

### Toast Component

**Arquivo:** `/app/frontend/src/Toast.js`

**Props:**
```javascript
{
  message: string,      // Texto a exibir
  type: string,         // 'success' | 'error' | 'warning' | 'info'
  duration: number,     // ms (0 = permanente)
  onClose: function     // Callback ao fechar
}
```

**Recursos:**
- Auto-close após duration
- Botão de fechar manual (×)
- Animação de slide-in
- Responsivo (mobile-friendly)

---

## 📊 Fluxo de Erro Completo

### Cenário: Rate Limit (429)

```
1. Usuário envia mensagem
   ↓
2. Backend detecta erro 429
   ↓
3. Backend extrai retry_delay (30s)
   ↓
4. Backend retorna:
   {
     error: "Muitas requisições...",
     retry_after: 30
   }
   ↓
5. Frontend recebe resposta
   ↓
6. showToast() exibe notificação
   ↓
7. startRetryTimer(30) inicia countdown
   ↓
8. UI atualiza:
   - Input: "Aguarde 30s..."
   - Botão: "⏱️ 30s"
   - Toast: Mensagem visível
   ↓
9. Countdown a cada 1s
   - 29s... 28s... 27s...
   ↓
10. Quando chega a 0:
    - Botão reabilitado
    - Input normal
    - Toast pode fechar
    ↓
11. Usuário pode tentar novamente
```

---

## 💡 Boas Práticas

### Para Desenvolvedores

1. **Use Toast para todos os feedbacks:**
```javascript
// ✅ BOM
showToast('Operação concluída', 'success');

// ❌ EVITE
alert('Operação concluída');
console.log('Operação concluída');
```

2. **Escolha o tipo correto:**
```javascript
// Sucesso
showToast('Arquivo processado!', 'success');

// Erro permanente
showToast('Arquivo não encontrado', 'error');

// Aviso temporário (rate limit)
showToast('Aguarde...', 'warning', 0);

// Informação
showToast('Processando em background', 'info');
```

3. **Durações apropriadas:**
```javascript
// Sucesso: 5 segundos
showToast('Concluído!', 'success', 5000);

// Erro: 5 segundos
showToast('Erro!', 'error', 5000);

// Rate limit: Permanente (0)
showToast('Aguarde...', 'warning', 0);
```

### Para Usuários

1. **Rate Limit apareceu?**
   - ✅ Aguarde o countdown terminar
   - ✅ Não force múltiplos cliques
   - ✅ Verifique se não está fazendo muitas requisições

2. **Erro persistente?**
   - 🔄 Recarregue a página
   - 🔍 Verifique a conexão
   - 📝 Veja os logs: `/var/log/supervisor/backend.err.log`

3. **Toast não fecha?**
   - Clique no × para fechar manualmente
   - Rate limits só fecham quando permitido

---

## 🐛 Troubleshooting

### Toast não aparece

**Causa:** Componente não importado ou estado não configurado

**Solução:**
```javascript
import Toast from './Toast';
const [toast, setToast] = useState(null);

// No render:
{toast && <Toast {...toast} onClose={closeToast} />}
```

### Countdown não funciona

**Causa:** setInterval não limpo corretamente

**Solução:**
```javascript
useEffect(() => {
  return () => {
    if (retryTimerRef.current) {
      clearInterval(retryTimerRef.current);
    }
  };
}, []);
```

### Retry delay não detectado

**Causa:** Backend não retornando retry_after

**Solução:** Verifique se o backend está usando regex correta:
```python
retry_match = re.search(r'retry in (\d+\.?\d*)', error_str)
```

### Mensagem em inglês aparece

**Causa:** Erro não mapeado para português

**Solução:** Adicione tratamento no backend:
```python
if "specific_error" in error_str:
    error_message = "Mensagem em português"
```

---

## 📈 Melhorias Futuras

### Planejadas:
- [ ] Retry automático após countdown
- [ ] Histórico de erros no console dev
- [ ] Métricas de rate limits
- [ ] Modo offline gracioso
- [ ] Toast empilháveis (múltiplos simultâneos)

### Em consideração:
- [ ] Som de notificação (opcional)
- [ ] Vibração em mobile
- [ ] Posição configurável do toast
- [ ] Temas dark/light para toast

---

## 🎉 Resumo dos Benefícios

| Antes | Agora |
|-------|-------|
| Erro técnico na tela | Mensagem amigável |
| App pode travar | Sempre responsivo |
| Sem feedback visual | Countdown visível |
| Usuário confuso | Ação clara |
| Alert bloqueante | Toast discreto |
| Sem tratamento 429 | Retry automático |
| Inglês técnico | Português claro |

---

## 📚 Arquivos Relacionados

- `/app/backend/server.py` - Tratamento de erros no backend
- `/app/frontend/src/App.js` - Lógica de retry e toast
- `/app/frontend/src/Toast.js` - Componente de notificação
- `/app/frontend/src/Toast.css` - Estilos do toast
- `/app/frontend/src/App.css` - Estilos do input rate-limited

---

**Sistema de tratamento de erros implementado com sucesso! 🎉**
