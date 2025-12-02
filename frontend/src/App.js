import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import Toast from './Toast';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

function App() {
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
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const retryTimerRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    checkSystemHealth();
    checkIndexStatus();
  }, []);

  useEffect(() => {
    // Cleanup retry timer on unmount
    return () => {
      if (retryTimerRef.current) {
        clearInterval(retryTimerRef.current);
      }
    };
  }, []);

  const showToast = (message, type = 'info', duration = 5000) => {
    setToast({ message, type, duration });
  };

  const closeToast = () => {
    setToast(null);
  };

  const startRetryTimer = (seconds) => {
    setIsRateLimited(true);
    setRetryTimer(seconds);

    if (retryTimerRef.current) {
      clearInterval(retryTimerRef.current);
    }

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
      console.error('Health check failed:', error);
      setSystemStatus({
        status: 'error',
        api_key_configured: false,
        database_loaded: false,
        message: 'Cannot connect to backend'
      });
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

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!input.trim() || loading || isRateLimited) return;

    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    const userInput = input;
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post(`${BACKEND_URL}/chat`, {
        message: userInput
      });

      // Check for rate limit error
      if (response.data.error && response.data.retry_after) {
        showToast(response.data.error, 'warning', 0);
        startRetryTimer(response.data.retry_after);
        
        // Remove the user message since we couldn't process it
        setMessages(prev => prev.slice(0, -1));
        // Restore the input
        setInput(userInput);
      } else if (response.data.error) {
        // Other errors
        showToast(response.data.error, 'error');
        
        const errorMessage = {
          role: 'assistant',
          content: response.data.error,
          error: true,
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, errorMessage]);
      } else {
        // Success
        const assistantMessage = {
          role: 'assistant',
          content: response.data.response,
          error: false,
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, assistantMessage]);
      }
    } catch (error) {
      // Network or other errors
      let errorMsg = 'Erro de conexão. Verifique sua internet e tente novamente.';
      
      if (error.response?.status === 429) {
        errorMsg = 'Muitas requisições no momento. Aguarde alguns segundos e tente novamente.';
        showToast(errorMsg, 'warning', 0);
        startRetryTimer(30);
        
        // Remove the user message and restore input
        setMessages(prev => prev.slice(0, -1));
        setInput(userInput);
      } else {
        showToast(errorMsg, 'error');
        
        const errorMessage = {
          role: 'assistant',
          content: errorMsg,
          error: true,
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  const handleFileUpload = async (event) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    setIngesting(true);
    setShowIngestionMenu(false);
    setIngestionMessage(`Processando ${files.length} arquivo(s) PDF...`);

    try {
      const formData = new FormData();
      for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
      }

      const response = await axios.post(`${BACKEND_URL}/ingest-pdf`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.status === 'success') {
        showToast(
          `${response.data.message}. ${response.data.chunks_created} fragmentos criados.`,
          'success'
        );
        setIngestionMessage('');
        checkSystemHealth();
        checkIndexStatus();
      } else {
        showToast(response.data.message, 'error');
        setIngestionMessage('');
      }
      
      setIngesting(false);

    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.message || 'Erro ao processar arquivos';
      showToast(errorMsg, 'error');
      setIngesting(false);
      setIngestionMessage('');
    }

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleUrlIngestion = async () => {
    if (!urlInput.trim()) {
      showToast('Por favor, insira uma URL válida', 'warning');
      return;
    }

    setIngesting(true);
    setShowUrlInput(false);
    setShowIngestionMenu(false);
    setIngestionMessage(`Processando URL: ${urlInput}...`);

    try {
      const formData = new FormData();
      formData.append('url', urlInput);

      const response = await axios.post(`${BACKEND_URL}/ingest-url`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.status === 'success') {
        showToast(
          `${response.data.message}. ${response.data.chunks_created} fragmentos criados.`,
          'success'
        );
        setIngestionMessage('');
        checkSystemHealth();
        checkIndexStatus();
      } else {
        showToast(response.data.message, 'error');
        setIngestionMessage('');
      }
      
      setIngesting(false);
      setUrlInput('');

    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.message || 'Erro ao processar URL';
      showToast(errorMsg, 'error');
      setIngesting(false);
      setIngestionMessage('');
    }
  };

  return (
    <div className="app">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="sidebar-header">
          <h2>🤖 GeneXus AI</h2>
          <p className="subtitle">Assistant</p>
        </div>

        <div className="sidebar-content">
          <div className="status-section">
            <h3>📊 Status</h3>
            {systemStatus && (
              <div className="status-info">
                <div className={`status-badge status-${systemStatus.status}`}>
                  {systemStatus.status === 'healthy' ? '✅ Online' : '⚠️ Degraded'}
                </div>
                <div className="status-details">
                  <div className="status-item">
                    <span>API Key:</span>
                    <span>{systemStatus.api_key_configured ? '✅' : '❌'}</span>
                  </div>
                  <div className="status-item">
                    <span>Database:</span>
                    <span>{systemStatus.database_loaded ? '✅' : '❌'}</span>
                  </div>
                </div>
                <p className="status-message">{systemStatus.message}</p>
              </div>
            )}
          </div>

          {indexStatus && indexStatus.exists && (
            <div className="index-section">
              <h3>📊 Index</h3>
              <p className="index-count">{indexStatus.document_count} chunks</p>
            </div>
          )}

          <div className="info-section">
            <h3>ℹ️ About</h3>
            <p className="info-text">
              This assistant uses RAG (Retrieval Augmented Generation) to search GeneXus documentation and provide specialized answers.
            </p>
            <div className="tech-stack">
              <div className="tech-item">🧠 Gemini 2.0 Flash</div>
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
              📚 Ingest Documents
            </button>
            
            {showIngestionMenu && !ingesting && (
              <div className="ingestion-menu">
                <button 
                  onClick={() => fileInputRef.current?.click()}
                  className="ingestion-option"
                  data-testid="ingest-pdf-button"
                >
                  📄 Upload PDF Files
                  <span className="option-desc">Select one or more PDF files</span>
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf"
                  multiple
                  style={{ display: 'none' }}
                  onChange={handleFileUpload}
                />
                <button 
                  onClick={() => {
                    setShowUrlInput(true);
                    setShowIngestionMenu(false);
                  }}
                  className="ingestion-option"
                  data-testid="ingest-url-button"
                >
                  🌐 From URL
                  <span className="option-desc">Enter a documentation URL</span>
                </button>
              </div>
            )}
            
            {showUrlInput && !ingesting && (
              <div className="url-input-container">
                <input
                  type="text"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  placeholder="https://docs.genexus.com/..."
                  className="url-input"
                  data-testid="url-input"
                />
                <div className="url-buttons">
                  <button 
                    onClick={handleUrlIngestion}
                    className="url-button url-button-submit"
                    disabled={!urlInput.trim()}
                  >
                    ✅ Ingest
                  </button>
                  <button 
                    onClick={() => {
                      setShowUrlInput(false);
                      setUrlInput('');
                    }}
                    className="url-button url-button-cancel"
                  >
                    ❌ Cancel
                  </button>
                </div>
              </div>
            )}
            
            {ingestionMessage && (
              <div className={`ingestion-status ${ingesting ? 'ingesting' : ''}`}>
                {ingestionMessage}
              </div>
            )}
          </div>

          {messages.length > 0 && (
            <button onClick={clearChat} className="clear-button">
              🗑️ Clear Chat
            </button>
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="main-content">
        <div className="chat-header">
          <h1>🤖 GeneXus AI Assistant</h1>
          <p className="header-subtitle">Specialized in GeneXus development powered by official documentation</p>
        </div>

        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="welcome-message">
              <div className="welcome-icon">👋</div>
              <h2>Welcome to GeneXus AI Assistant!</h2>
              <p>Ask me anything about GeneXus development, and I'll search the documentation to help you.</p>
              <div className="example-questions">
                <p className="example-label">Try asking:</p>
                <div className="example-item">"Como criar um Data Provider em GeneXus?"</div>
                <div className="example-item">"What are the best practices for GeneXus objects?"</div>
                <div className="example-item">"Explain GeneXus transactions"</div>
              </div>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div key={index} className={`message message-${msg.role} ${msg.error ? 'message-error' : ''}`}>
                <div className="message-icon">
                  {msg.role === 'user' ? '👤' : msg.error ? '❌' : '🤖'}
                </div>
                <div className="message-content">
                  <div className="message-text">{msg.content}</div>
                  <div className="message-time">
                    {new Date(msg.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))
          )}
          {loading && (
            <div className="message message-assistant">
              <div className="message-icon">🤖</div>
              <div className="message-content">
                <div className="loading-indicator">
                  <div className="loading-dot"></div>
                  <div className="loading-dot"></div>
                  <div className="loading-dot"></div>
                  <span className="loading-text">Thinking...</span>
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
              placeholder={isRateLimited ? `Aguarde ${retryTimer}s...` : "Pergunte algo sobre GeneXus..."}
              className="chat-input"
              disabled={loading || isRateLimited}
              data-testid="chat-input"
            />
            <button 
              type="submit" 
              className="send-button" 
              disabled={loading || !input.trim() || isRateLimited}
              data-testid="send-button"
              title={isRateLimited ? `Aguarde ${retryTimer} segundos` : 'Enviar mensagem'}
            >
              {loading ? '⏳' : isRateLimited ? `⏱️ ${retryTimer}s` : '🚀'}
            </button>
          </form>
        </div>
      </div>

      {/* Toast Notifications */}
      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          duration={toast.duration}
          onClose={closeToast}
        />
      )}
    </div>
  );
}

export default App;
