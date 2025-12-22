import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import Toast from './Toast';
import LogsViewer from './LogsViewer';
import KnowledgeManager from './KnowledgeManager';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

// Função auxiliar simples para gerar UUID
const generateUUID = () => {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    var r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
};

function App() {
  // --- STATE INICIAL ---
  const [config, setConfig] = useState({
    app_name: "AI Assistant", 
    app_subtitle: "Connecting to server...",
    welcome_message: "Please wait...",
    primary_color: "#333",
    secondary_color: "#555",
    logo_emoji: "⏳"
  });

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [systemStatus, setSystemStatus] = useState(null);
  const [indexStatus, setIndexStatus] = useState(null);
  const [ingesting, setIngesting] = useState(false);
  const [ingestionMessage, setIngestionMessage] = useState('');
  const [showIngestionMenu, setShowIngestionMenu] = useState(false);
  const [showUrlInput, setShowUrlInput] = useState(false);
  const [urlInput, setUrlInput] = useState('');
  const [toast, setToast] = useState(null);
  const [isRateLimited, setIsRateLimited] = useState(false);
  const [retryTimer, setRetryTimer] = useState(0);
  const [showLogs, setShowLogs] = useState(false);
  const [showManager, setShowManager] = useState(false);

  // --- SESSION ID (PERSISTENTE) ---
  // Tenta pegar do localStorage, se não existir cria um e salva
  const [sessionId, setSessionId] = useState(() => {
    const saved = localStorage.getItem('chat_session_id');
    if (saved) return saved;
    const newId = generateUUID();
    localStorage.setItem('chat_session_id', newId);
    return newId;
  });
  
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const retryTimerRef = useRef(null);

  // --- EFEITO INICIAL ROBUSTO ---
  useEffect(() => {
    // LOG DE DEBUG PARA VERIFICAR A SESSÃO
    console.log("🔍 Current Session ID:", sessionId);

    const fetchConfig = async () => {
      try {
        console.log(`Tentando conectar em: ${BACKEND_URL}/api/config`);
        const response = await axios.get(`${BACKEND_URL}/api/config`);
        
        // SUCESSO: Usa config do backend
        const newConfig = response.data;
        setConfig(newConfig);
        applyTheme(newConfig);
        document.title = newConfig.app_name;
        fetchHistory(sessionId);
        
      } catch (error) {
        console.error("Erro fatal ao carregar config:", error);
        
        // FALHA: Usa Configuração de Fallback (Para a tela não travar)
        const fallbackConfig = {
            app_name: "GeneXus AI (Offline)",
            app_subtitle: "Backend Connection Failed",
            welcome_message: "⚠️ Could not connect to the backend server. Please check if main.py is running.",
            primary_color: "#666666",
            secondary_color: "#888888",
            logo_emoji: "🔌"
        };
        
        setConfig(fallbackConfig);
        applyTheme(fallbackConfig);
        
        showToast(`Connection Error: ${error.message}. Is backend running on port 8001?`, "error");
      }
    };

    fetchConfig();
    checkSystemHealth();
    checkIndexStatus();
  }, [sessionId]); // Adicionado sessionId como dependência para garantir que loga o valor correto

  const fetchHistory = async (sid) => {
    try {
        const res = await axios.get(`${BACKEND_URL}/api/chat/history/${sid}`);
        if (res.data.history && res.data.history.length > 0) {
            setMessages(res.data.history);
            showToast("Chat history restored", "success", 2000);
        }
    } catch (error) {
        console.error("Failed to load history", error);
    }
  };

  const applyTheme = (themeConfig) => {
    const root = document.documentElement;
    if (themeConfig.primary_color) {
      root.style.setProperty('--primary-color', themeConfig.primary_color);
    }
    if (themeConfig.secondary_color) {
      root.style.setProperty('--secondary-color', themeConfig.secondary_color);
    }
  };

  const scrollToBottom = () => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); };
  useEffect(() => { scrollToBottom(); }, [messages]);
  useEffect(() => { return () => { if (retryTimerRef.current) clearInterval(retryTimerRef.current); }; }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    return () => {
      if (retryTimerRef.current) clearInterval(retryTimerRef.current);
    };
  }, []);

  const showToast = (message, type = 'info', duration = 5000) => {
    setToast({ message, type, duration });
  };

  const closeToast = () => setToast(null);

  const startRetryTimer = (seconds) => {
    setIsRateLimited(true);
    setRetryTimer(seconds);
    if (retryTimerRef.current) clearInterval(retryTimerRef.current);
    retryTimerRef.current = setInterval(() => {
      setRetryTimer((prev) => {
        if (prev <= 1) {
          clearInterval(retryTimerRef.current);
          setIsRateLimited(false);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const checkSystemHealth = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/health`);
      setSystemStatus(response.data);
    } catch (error) {
      setSystemStatus({ status: 'error', message: 'Backend Offline' });
    }
  };

  const checkIndexStatus = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/index-status`);
      setIndexStatus(response.data);
    } catch (error) {
      console.error('Index check failed:', error);
    }
  };

  // --- NOVA FUNÇÃO: COPIAR SESSION ID ---
  const handleCopySessionId = () => {
    navigator.clipboard.writeText(sessionId);
    showToast("Session ID copied to clipboard! 📋", "success", 2000);
  };

  // Função para criar nova sessão (Limpar memória real)
  const handleNewSession = () => {
    const newId = generateUUID();
    setSessionId(newId);
    localStorage.setItem('chat_session_id', newId);
    setMessages([]); // Limpa a tela
    showToast("Started new conversation context", "info");         
  };

  // --- NOVA FUNÇÃO: COPIAR MENSAGEM ---
  const handleCopyMessage = (text) => {
    navigator.clipboard.writeText(text);
    showToast("Markdown copied to clipboard! 📋", "success", 2000);
  };

  // --- FUNÇÃO DE FEEDBACK ---
  const handleFeedback = async (index, score) => {
    const message = messages[index];
    const prevMessage = messages[index - 1]; // Assume que a anterior é a pergunta do usuário

    if (!message || !prevMessage) return;

    // Atualiza UI Otimisticamente (Marca como votado)
    const updatedMessages = [...messages];
    updatedMessages[index] = { ...message, feedback: score };
    setMessages(updatedMessages);

                                                                                          
    try {
      await axios.post(`${BACKEND_URL}/api/feedback/`, {
        user_question: prevMessage.content,
        bot_response: message.content,
        score: score,
        comment: "" 
      });
      if (score > 0) showToast("Obrigado pelo feedback positivo! 👍", "success", 2000);
      else showToast("Obrigado! Vamos melhorar com seu feedback. 👎", "info", 2000);
    } catch (error) {
      console.error("Feedback error:", error);
      
      // Tratamento de erro detalhado para Debug
      const status = error.response?.status;
      const detail = error.response?.data?.detail;
      const errorMsg = status ? `Erro ${status}: ${detail || 'Falha no servidor'}` : error.message;
      
      showToast(`Falha ao enviar feedback: ${errorMsg}`, "error");

      // Reverte o estado visual se falhar (para o usuário poder tentar de novo)
      const revertedMessages = [...messages];
      revertedMessages[index] = { ...message, feedback: null };
      setMessages(revertedMessages);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading || isRateLimited) return;

    const userMessage = { role: 'user', content: input, timestamp: new Date().toISOString() };
    setMessages(prev => [...prev, userMessage]);
    const userInput = input;
    setInput('');
    setLoading(true);

    try {
      // 1. Cria a mensagem do assistente vazia inicialmente
      const assistantMsgId = new Date().getTime(); // ID temporário
      setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: '', 
          timestamp: new Date().toISOString(),
          id: assistantMsgId
      }]);

      // 2. Inicia o Fetch
      const response = await fetch(`${BACKEND_URL}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userInput, session_id: sessionId })
                                                            
      });

      if (!response.ok) {
        if (response.status === 429) throw new Error("Rate limit exceeded");
        throw new Error("Network response was not ok");
      }

      // 3. Lê o Stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = ""; // Buffer para texto acumulado
      let sources = []; // Buffer para fontes

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        // O stream pode mandar pedaços de JSON quebrados, ou múltiplos JSONs na mesma linha se for muito rápido.
        // A nossa API manda NDJSON (um JSON por linha).
        const lines = chunk.split('\n');
        
        for (const line of lines) {
            if (!line.trim()) continue;
            try {
                const data = JSON.parse(line);
                
                if (data.type === 'token') {
                    buffer += data.content;
                    // Atualiza o estado com o texto acumulado em tempo real
                    setMessages(prev => {
                        const newMsgs = [...prev];
                        const lastMsg = newMsgs[newMsgs.length - 1];
                        if (lastMsg.role === 'assistant') {
                            lastMsg.content = buffer;
                        }
                        return newMsgs;
                    });
                } else if (data.type === 'sources') {
                    sources = data.content;
                } else if (data.type === 'error') {
                    throw new Error(data.content);
                }
            } catch (e) {
                // Se der erro de parse (linha incompleta), ignora e espera o resto no próximo chunk
                // Em NDJSON robusto precisaria de buffer de linha, mas para este demo simples ok.
            }
        }
      }

      // 4. Finalização: Adiciona fontes ao texto se houver
      if (sources.length > 0) {
          const footer = "\n\n---\n📚 **Fontes Consultadas:**\n" + sources.map(s => `- \`${s}\``).join('\n');
          setMessages(prev => {
              const newMsgs = [...prev];
              const lastMsg = newMsgs[newMsgs.length - 1];
              lastMsg.content = buffer + footer; // Garante formatação final
              return newMsgs;
          });
      }

    } catch (error) {
      let errorMsg = 'Connection error.';
                                           
      if (error.message.includes('Rate limit')) errorMsg = 'Muitas requisições. Aguarde um pouco.';
      
      setMessages(prev => {
          // Remove a mensagem vazia ou substitui por erro
          const newMsgs = [...prev];
          const lastMsg = newMsgs[newMsgs.length - 1];
          if (lastMsg.role === 'assistant') {
              lastMsg.content = errorMsg;
              lastMsg.error = true;
          }
          return newMsgs;
      });
      showToast(errorMsg, 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (event) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    setIngesting(true);
    setShowIngestionMenu(false);
    setIngestionMessage(`Uploading ${files.length} PDF(s)...`);

    try {
      const formData = new FormData();
      for (let i = 0; i < files.length; i++) formData.append('files', files[i]);

      const response = await axios.post(`${BACKEND_URL}/api/ingest-pdf`, formData, { headers: { 'Content-Type': 'multipart/form-data' } });
      
      if (response.data.status === 'success') {
        showToast(`${response.data.message}`, 'success');
        checkSystemHealth();
        checkIndexStatus();
      } else {
        showToast(response.data.message, 'error');
      }
    } catch (error) {
      showToast('Upload failed', 'error');
    } finally {
      setIngesting(false);
      setIngestionMessage('');
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleUrlIngestion = async () => {
    if (!urlInput.trim()) return;
    setIngesting(true);
    setShowUrlInput(false);
    setIngestionMessage(`Processing URL...`);

    try {
      const formData = new FormData();
      formData.append('url', urlInput);
      const response = await axios.post(`${BACKEND_URL}/api/ingest-url`, formData);
      
      if (response.data.status === 'success') {
        showToast('URL Ingested successfully', 'success');
        checkSystemHealth();
        checkIndexStatus();
      } else {
        showToast(response.data.message, 'error');
      }
    } catch (error) {
      showToast('URL Ingest failed', 'error');
    } finally {
      setIngesting(false);
      setIngestionMessage('');
      setUrlInput('');
    }
  };

  return (
    <div className="app">
      <div className="sidebar">
        <div className="sidebar-header">
          <h2>{config.logo_emoji} {config.app_name}</h2>
          <p className="subtitle">{config.app_subtitle}</p>
        </div>

        <div className="sidebar-content">
          <div className="status-section">
            <h3>📊 Status</h3>
            {systemStatus && (
              <div className="status-info">
                <div className={`status-badge status-${systemStatus.status}`}>
                  {systemStatus.status === 'healthy' ? '✅ Online' : '⚠️ Degraded'}
                </div>
                {systemStatus.app_name && 
                  <p className="option-desc" style={{marginBottom:'8px'}}>Instance: {systemStatus.app_name}</p>
                }
                <div className="status-details">
                  <div className="status-item"><span>API Key:</span><span>{systemStatus.api_key_configured ? '✅' : '❌'}</span></div>
                  <div className="status-item"><span>Database:</span><span>{systemStatus.database_loaded ? '✅' : '❌'}</span></div>
                  <div 
                    className="status-item" 
                    onClick={handleCopySessionId} 
                    style={{cursor: 'pointer'}} 
                    title="Click to copy Session ID"
                  >
                    <span>Session:</span>
                    <span style={{fontSize: '11px', fontFamily: 'monospace', textDecoration: 'underline dotted'}}>
                      {sessionId.slice(0, 6)}...
                    </span>
                  </div>
                </div>
                <p className="status-message">{systemStatus.message}</p>
              </div>
            )}
          </div>

          {indexStatus && indexStatus.exists && (
            <div className="index-section">
              <h3>📊 Knowledge Base</h3>
              <p className="index-count">{indexStatus.document_count} chunks</p>
              {indexStatus.collection_name && 
                <p className="option-desc">Collection: {indexStatus.collection_name}</p>
              }
              <button onClick={() => setShowManager(true)} className="manage-link">Manage Files ⚙️</button>
            </div>
          )}

          <div className="info-section">
            <h3>ℹ️ About</h3>
            <p className="info-text">
              This assistant uses RAG (Retrieval Augmented Generation) to answer questions based on the provided documentation.
            </p>
            <div className="tech-stack">
              <div className="tech-item">🧠 {config.llm?.model_name || "Gemini"}</div>
              <div className="tech-item">📚 ChromaDB</div>
              <div className="tech-item">🔗 LangChain</div>
            </div>
          </div>

          <div className="actions-section">
            <button 
              onClick={() => setShowIngestionMenu(!showIngestionMenu)} 
              className="action-button ingestion-button"
              disabled={ingesting}
            >
              📚 Add Knowledge
            </button>
            
            {showIngestionMenu && !ingesting && (
              <div className="ingestion-menu">
                <button onClick={() => fileInputRef.current?.click()} className="ingestion-option">
                  📄 Upload PDF Files
                </button>
                <input ref={fileInputRef} type="file" accept=".pdf" multiple style={{ display: 'none' }} onChange={handleFileUpload} />
                <button onClick={() => { setShowUrlInput(true); setShowIngestionMenu(false); }} className="ingestion-option">
                  🌐 From URL
                </button>
              </div>
            )}
            
            {showUrlInput && !ingesting && (
              <div className="url-input-container">
                <input type="text" value={urlInput} onChange={(e) => setUrlInput(e.target.value)} placeholder="https://..." className="url-input" />
                <div className="url-buttons">
                  <button onClick={handleUrlIngestion} className="url-button url-button-submit" disabled={!urlInput.trim()}>Add</button>
                  <button onClick={() => setShowUrlInput(false)} className="url-button url-button-cancel">Cancel</button>
                </div>
              </div>
            )}
            
            {ingestionMessage && <div className={`ingestion-status ${ingesting ? 'ingesting' : ''}`}>{ingestionMessage}</div>}
          </div>

          <div className="actions-section">
            <button onClick={() => setShowLogs(true)} className="action-button logs-button">
              📊 System Logs
            </button>
          </div>
          <button onClick={handleNewSession} className="clear-button">✨ New Chat</button>

          {messages.length > 0 && <button onClick={() => setMessages([])} className="clear-button">🗑️ Clear Chat</button>}
        </div>
      </div>

      <div className="main-content">
        <div className="chat-header">
          <h1>{config.logo_emoji} {config.app_name}</h1>
          <p className="header-subtitle">{config.app_subtitle}</p>
        </div>

        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="welcome-message">
              <div className="welcome-icon">{config.logo_emoji}</div>
              <h2>Hello!</h2>
              <p>{config.welcome_message}</p>
              <div className="example-questions">
                <p className="example-label">Tente perguntar:</p>
                <div className="example-item" onClick={() => setInput("O que é Genexus?")}>"O que é Genexus?"</div>
                <div className="example-item" onClick={() => setInput("Crie um exemplo de transação")}>"Crie um exemplo de transação"</div>
              </div>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div key={index} className={`message-group ${msg.role}`}>
                <div className={`message message-${msg.role} ${msg.error ? 'message-error' : ''}`}>
                  <div className="message-icon">{msg.role === 'user' ? '👤' : msg.error ? '❌' : config.logo_emoji}</div>
                  <div className="message-content">
                    <div className="message-text">
                      <ReactMarkdown 
                        components={{         
                          a: ({node, ...props}) => <a {...props} target="_blank" rel="noopener noreferrer" />                                                                                                                                                                                                                           
                        }}>{msg.content}</ReactMarkdown>
                    </div>
                    <div className="message-footer">
                        <span className="message-time">{new Date(msg.timestamp).toLocaleTimeString()}</span>
                        
                        {/* --- FEEDBACK BUTTONS --- */}
                        {msg.role === 'assistant' && !msg.error && (
                            <div className="feedback-actions">
                                <button 
                                  className="feedback-btn" 
                                  onClick={() => handleCopyMessage(msg.content)} 
                                  title="Copy Markdown"
                                >
                                  📋
                                </button>
                                <button 
                                    className={`feedback-btn ${msg.feedbackGiven === 1 ? 'active' : ''}`}
                                    onClick={() => handleFeedback(index, 1)}
                                    disabled={msg.feedbackGiven}
                                    title="Good response"
                                >
                                    👍
                                </button>
                                <button 
                                    className={`feedback-btn ${msg.feedbackGiven === -1 ? 'active' : ''}`}
                                    onClick={() => handleFeedback(index, -1)}
                                    disabled={msg.feedbackGiven}
                                    title="Bad response"
                                >
                                    👎
                                </button>
                            </div>
                        )}
                    </div>
                  </div>
                </div>
              </div>
            ))
          )}
          {loading && (
            <div className="message message-assistant">
              <div className="message-icon">{config.logo_emoji}</div>
              <div className="message-content">
                <div className="loading-indicator">
                  <div className="loading-dot"></div><div className="loading-dot"></div><div className="loading-dot"></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-container">
          <form onSubmit={handleSubmit} className="input-form">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={isRateLimited ? `Wait ${retryTimer}s...` : `Ask ${config.app_name}...`}
              className="chat-input"
              disabled={loading || isRateLimited}
            />
            <button type="submit" className="send-button" disabled={loading || !input.trim() || isRateLimited}>
              {loading ? '⏳' : isRateLimited ? `⏱️` : '🚀'}
            </button>
          </form>
        </div>
      </div>

      {toast && <Toast message={toast.message} type={toast.type} duration={toast.duration} onClose={closeToast} />}
      {showLogs && <LogsViewer onClose={() => setShowLogs(false)} />}
      {showManager && <KnowledgeManager onClose={() => { setShowManager(false); checkIndexStatus(); }} backendUrl={BACKEND_URL} />}
    </div>
  );
}

export default App;