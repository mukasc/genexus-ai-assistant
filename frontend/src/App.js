import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import Toast from './Toast';
import LogsViewer from './LogsViewer';
import KnowledgeManager from './KnowledgeManager';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

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
    app_subtitle: "Connecting...",
    welcome_message: "Please wait...",
    primary_color: "#333", secondary_color: "#555", logo_emoji: "⏳"
  });

  const [profiles, setProfiles] = useState([]); 
  const [activeProfile, setActiveProfile] = useState("");

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [systemStatus, setSystemStatus] = useState(null);
  const [indexStatus, setIndexStatus] = useState(null);
  const [ingesting, setIngesting] = useState(false);
  const [ingestionMessage, setIngestionMessage] = useState('');
  const [showIngestionMenu, setShowIngestionMenu] = useState(false);
  const [showUrlInput, setShowUrlInput] = useState(false);
  const [showYoutubeInput, setShowYoutubeInput] = useState(false);
  const [urlInput, setUrlInput] = useState('');
  const [toast, setToast] = useState(null);
  const [isRateLimited, setIsRateLimited] = useState(false);
  const [retryTimer, setRetryTimer] = useState(0);
  const [showLogs, setShowLogs] = useState(false);
  const [showManager, setShowManager] = useState(false);
  
  // CORREÇÃO: Inicia null para evitar carregar sessão "genérica" antes da hora
  const [sessionId, setSessionId] = useState(null);

  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const retryTimerRef = useRef(null);

  // --- 1. EFEITO DE INICIALIZAÇÃO (Roda APENAS 1 vez) ---
  useEffect(() => {
    console.log("🚀 App Mounted. Initializing...");
    fetchInitialData();
    // eslint-disable-next-line
  }, []); 

  // --- 2. GERENCIADOR DE PERFIL ---
  // Quando o perfil muda, carregamos a sessão correspondente a ele
  useEffect(() => {
    if (activeProfile) {
        console.log(`👤 Active Profile changed to: ${activeProfile}`);
        loadSessionForProfile(activeProfile);
    }
  }, [activeProfile]);

  // --- 3. GERENCIADOR DE HISTÓRICO ---
  // Quando o ID da sessão muda, buscamos o histórico e atualizamos status
  useEffect(() => {
    if (sessionId) {
        console.log(`🔄 Session ID changed to: ${sessionId}. Fetching history...`);
        setMessages([]); // Limpa a tela antes de carregar o novo histórico
        fetchHistory(sessionId);
        checkSystemHealth();
        checkIndexStatus();
    }
  }, [sessionId]);

  const fetchInitialData = async () => {
    try {
        console.log(`Tentando conectar em: ${BACKEND_URL}/api/config`);
        const configRes = await axios.get(`${BACKEND_URL}/api/config`);
        setConfig(configRes.data);
        applyTheme(configRes.data);
        document.title = configRes.data.app_name;

        try {
            const profilesRes = await axios.get(`${BACKEND_URL}/api/config/profiles`);
            setProfiles(profilesRes.data.profiles || []);
            // Define o perfil ativo (Isso dispara o useEffect[2])
            setActiveProfile(profilesRes.data.active || "");
        } catch (e) {
            console.warn("Endpoint de perfis não disponível:", e);
        }
        
    } catch (error) {
        console.error("Erro fatal:", error);
        const fallbackConfig = {
            app_name: "GeneXus AI (Offline)",
            app_subtitle: "Backend Connection Failed",
            welcome_message: "⚠️ Could not connect to backend.",
            primary_color: "#666666", secondary_color: "#888888", logo_emoji: "🔌"
        };
        setConfig(fallbackConfig);
        applyTheme(fallbackConfig);
        showToast(`Connection Error: ${error.message}`, "error");
    }
  };

  const loadSessionForProfile = (profileName) => {
      const storageKey = `chat_session_${profileName}`;
      let savedId = localStorage.getItem(storageKey);
      if (!savedId) {
          savedId = generateUUID();
          localStorage.setItem(storageKey, savedId);
      }
      // Só atualiza se for diferente para evitar loop
      setSessionId(prev => (prev !== savedId ? savedId : prev));
  };

  const handleProfileSwitch = async (e) => {
      const newProfile = e.target.value;
      if (!newProfile || newProfile === activeProfile) return;
      
      try {
          // 1. Feedback imediato
          setMessages([]);
          
          // 2. Avisa backend
          await axios.post(`${BACKEND_URL}/api/config/switch`, { profile_id: newProfile });
          showToast(`Switched to ${newProfile}`, "success");
          
          // 3. Atualiza state local (Isso dispara a cadeia de efeitos: useEffect[2] -> loadSession -> useEffect[3] -> fetchHistory)
          // CORREÇÃO: Removemos a chamada manual loadSessionForProfile daqui para não conflitar com o useEffect
          setActiveProfile(newProfile);
          
          // 4. Config visual
          const configRes = await axios.get(`${BACKEND_URL}/api/config`);
          setConfig(configRes.data);
          applyTheme(configRes.data);
      } catch (err) {
          showToast("Failed to switch profile", "error");
      }
  };

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
    if (themeConfig.primary_color) root.style.setProperty('--primary-color', themeConfig.primary_color);
    if (themeConfig.secondary_color) root.style.setProperty('--secondary-color', themeConfig.secondary_color);
  };

  const scrollToBottom = () => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); };
  useEffect(() => { scrollToBottom(); }, [messages]);
  useEffect(() => { return () => { if (retryTimerRef.current) clearInterval(retryTimerRef.current); }; }, []);

  const showToast = (message, type = 'info', duration = 5000) => { setToast({ message, type, duration }); };
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

  const handleCopySessionId = () => {
    navigator.clipboard.writeText(sessionId);
    showToast("Session ID copied!", "success", 2000);
  };

  const handleNewSession = () => {
    const newId = generateUUID();
    setSessionId(newId);
    // Salva na chave específica do perfil ATUAL
    localStorage.setItem(`chat_session_${activeProfile}`, newId);
    setMessages([]); 
    showToast(`New conversation started for ${activeProfile}`, "info");         
  };

  const handleCopyMessage = (text) => {
    navigator.clipboard.writeText(text);
    showToast("Markdown copied to clipboard! 📋", "success", 2000);
  };

  const handleFeedback = async (index, score) => {
    const message = messages[index]; const prevMessage = messages[index - 1]; 
    if (!message || !prevMessage) return;
    const updatedMessages = [...messages]; updatedMessages[index] = { ...message, feedback: score }; setMessages(updatedMessages);
    try { await axios.post(`${BACKEND_URL}/api/feedback/`, { user_question: prevMessage.content, bot_response: message.content, score: score, comment: "" }); if (score > 0) showToast("Thanks for feedback! 👍", "success", 2000); else showToast("Thanks! We'll improve. 👎", "info", 2000); } 
    catch { showToast("Failed to send feedback", "error"); }
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
      const assistantMsgId = new Date().getTime(); 
      setMessages(prev => [...prev, { role: 'assistant', content: '', timestamp: new Date().toISOString(), id: assistantMsgId }]);

      const response = await fetch(`${BACKEND_URL}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userInput, session_id: sessionId })
      });

      if (!response.ok) {
        if (response.status === 429) throw new Error("Rate limit exceeded");
        throw new Error("Network response was not ok");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = ""; 
      let sources = []; 

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        for (const line of lines) {
            if (!line.trim()) continue;
            try {
                const data = JSON.parse(line);
                if (data.type === 'token') {
                    buffer += data.content;
                    setMessages(prev => {
                        const newMsgs = [...prev];
                        const lastMsg = newMsgs[newMsgs.length - 1];
                        if (lastMsg.role === 'assistant') lastMsg.content = buffer;
                        return newMsgs;
                    });
                } else if (data.type === 'sources') { sources = data.content; } 
                else if (data.type === 'error') { throw new Error(data.content); }
            } catch (e) {}
        }
      }

      if (sources.length > 0) {
          const footer = "\n\n---\n📚 **Fontes Consultadas:**\n" + sources.map(s => `- \`${s}\``).join('\n');
          setMessages(prev => {
              const newMsgs = [...prev];
              const lastMsg = newMsgs[newMsgs.length - 1];
              lastMsg.content = buffer + footer; 
              return newMsgs;
          });
      }
    } catch (error) {
      let errorMsg = 'Connection error.';
      if (error.message.includes('Rate limit')) errorMsg = 'Muitas requisições. Aguarde um pouco.';
      setMessages(prev => {
          const newMsgs = [...prev];
          const lastMsg = newMsgs[newMsgs.length - 1];
          if (lastMsg.role === 'assistant') { lastMsg.content = errorMsg; lastMsg.error = true; }
          return newMsgs;
      });
      showToast(errorMsg, 'error');
    } finally { setLoading(false); }
  };

  const handleFileUpload = async (event) => {
    const files = event.target.files; if (!files || files.length === 0) return;
    setIngesting(true); setShowIngestionMenu(false); setIngestionMessage(`Uploading ${files.length} PDF(s)...`);
    try {
      const formData = new FormData(); for (let i = 0; i < files.length; i++) formData.append('files', files[i]);
      const response = await axios.post(`${BACKEND_URL}/api/ingest-pdf`, formData, { headers: { 'Content-Type': 'multipart/form-data' } });
      if (response.data.status === 'success') { showToast(`${response.data.message}`, 'success'); checkSystemHealth(); checkIndexStatus(); } else { showToast(response.data.message, 'error'); }
    } catch (error) { showToast('Upload failed', 'error'); } finally { setIngesting(false); setIngestionMessage(''); if (fileInputRef.current) fileInputRef.current.value = ''; }
  };

  const handleUrlIngestion = async () => {
    if (!urlInput.trim()) return; setIngesting(true); setShowUrlInput(false); setIngestionMessage(`Processing URL...`);
    try {
      const formData = new FormData(); formData.append('url', urlInput);
      const response = await axios.post(`${BACKEND_URL}/api/ingest-url`, formData);
      if (response.data.status === 'success') { showToast('URL Ingested successfully', 'success'); checkSystemHealth(); checkIndexStatus(); } else { showToast(response.data.message, 'error'); }
    } catch (error) { showToast('URL Ingest failed', 'error'); } finally { setIngesting(false); setIngestionMessage(''); setUrlInput(''); }
  };

  const handleYoutubeIngestion = async () => {
    if (!urlInput.trim()) return; 
    setIngesting(true); setShowYoutubeInput(false); setIngestionMessage(`Transcribing Video...`);
    try {
      const formData = new FormData(); formData.append('url', urlInput);
      const response = await axios.post(`${BACKEND_URL}/api/ingest-youtube`, formData);
      if (response.data.status === 'success') { showToast(response.data.message, 'success'); checkSystemHealth(); checkIndexStatus(); } else { showToast(response.data.message, 'error'); }
    } catch (error) { showToast('Video Ingest failed. Check if video has subtitles.', 'error'); } finally { setIngesting(false); setIngestionMessage(''); setUrlInput(''); }
  };

  return (
    <div className="app">
      <div className="sidebar">
        <div className="sidebar-header">
          <h2>{config.logo_emoji} {config.app_name}</h2>
          <p className="subtitle">{config.app_subtitle}</p>
        </div>

        <div className="sidebar-content">
          
          {/* SELETOR DE PERFIL */}
          {profiles.length > 0 && (
            <div className="profile-selector">
                <label>Active Profile:</label>
                <select value={activeProfile} onChange={handleProfileSwitch} className="profile-dropdown">
                    {profiles.map(p => (
                        <option key={p} value={p}>{p.toUpperCase()}</option>
                    ))}
                </select>
            </div>
          )}

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
                  <div className="status-item"><span>DB:</span><span>{systemStatus.database_loaded ? '✅' : '❌'}</span></div>
                  <div className="status-item" onClick={handleCopySessionId} style={{cursor: 'pointer'}} title="Click to copy Session ID">
                    <span>Session:</span>
                    <span style={{fontSize: '11px', fontFamily: 'monospace', textDecoration: 'underline dotted'}}>
                      {sessionId ? sessionId.slice(0, 6) : '...'}...
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
            <button onClick={() => setShowIngestionMenu(!showIngestionMenu)} className="action-button ingestion-button" disabled={ingesting}>📚 Add Knowledge</button>
            
            {showIngestionMenu && !ingesting && (
              <div className="ingestion-menu">
                <button onClick={() => fileInputRef.current?.click()} className="ingestion-option">📄 PDF File</button>
                <input ref={fileInputRef} type="file" accept=".pdf" multiple style={{ display: 'none' }} onChange={handleFileUpload} />
                
                <button onClick={() => { setShowUrlInput(true); setShowIngestionMenu(false); }} className="ingestion-option">🌐 Website URL</button>
                
                <button onClick={() => { setShowYoutubeInput(true); setShowIngestionMenu(false); }} className="ingestion-option">📺 YouTube Video</button>
              </div>
            )}

            {showUrlInput && !ingesting && (
              <div className="url-input-container">
                <input type="text" value={urlInput} onChange={(e) => setUrlInput(e.target.value)} placeholder="https://website.com..." className="url-input" />
                <div className="url-buttons">
                  <button onClick={handleUrlIngestion} className="url-button url-button-submit">Add Site</button>
                  <button onClick={() => setShowUrlInput(false)} className="url-button url-button-cancel">Cancel</button>
                </div>
              </div>
            )}

            {showYoutubeInput && !ingesting && (
              <div className="url-input-container">
                <input type="text" value={urlInput} onChange={(e) => setUrlInput(e.target.value)} placeholder="https://youtube.com/watch?v=..." className="url-input" />
                <div className="url-buttons">
                  <button onClick={handleYoutubeIngestion} className="url-button url-button-submit" style={{backgroundColor: '#FF0000'}}>Add Video</button>
                  <button onClick={() => setShowYoutubeInput(false)} className="url-button url-button-cancel">Cancel</button>
                </div>
              </div>
            )}

            {ingestionMessage && <div className={`ingestion-status ${ingesting ? 'ingesting' : ''}`}>{ingestionMessage}</div>}
          </div>

          <div className="actions-section">
            <button onClick={() => setShowLogs(true)} className="action-button logs-button">📊 System Logs</button>
          </div>
          <button onClick={handleNewSession} className="clear-button">✨ New Chat</button>
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
                    {msg.role === 'assistant' && !msg.error && (
                      <div className="message-footer">
                        <span className="message-time">{new Date(msg.timestamp).toLocaleTimeString()}</span>
                        <div className="feedback-actions">
                          <button className="feedback-btn" onClick={() => handleCopyMessage(msg.content)} title="Copy Markdown">📋</button>
                          <button className={`feedback-btn ${msg.feedback === 1 ? 'active-like' : ''}`} onClick={() => handleFeedback(index, 1)} disabled={msg.feedback !== undefined && msg.feedback !== null}>👍</button>
                          <button className={`feedback-btn ${msg.feedback === -1 ? 'active-dislike' : ''}`} onClick={() => handleFeedback(index, -1)} disabled={msg.feedback !== undefined && msg.feedback !== null}>👎</button>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))
          )}
          {loading && (
            <div className="message message-assistant">
              <div className="message-icon">{config.logo_emoji}</div>
              <div className="message-content">
                <div className="loading-indicator"><div className="loading-dot"></div><div className="loading-dot"></div><div className="loading-dot"></div></div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-container">
          <form onSubmit={handleSubmit} className="input-form">
            <input type="text" value={input} onChange={(e) => setInput(e.target.value)} placeholder={isRateLimited ? `Wait ${retryTimer}s...` : `Ask ${config.app_name}...`} className="chat-input" disabled={loading || isRateLimited} />
            <button type="submit" className="send-button" disabled={loading || !input.trim() || isRateLimited}>{loading ? '⏳' : isRateLimited ? `⏱️` : '🚀'}</button>
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