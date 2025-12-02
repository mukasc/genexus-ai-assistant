import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
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
  const messagesEndRef = useRef(null);

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
    
    if (!input.trim() || loading) return;

    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post(`${BACKEND_URL}/api/chat`, {
        message: input
      });

      const assistantMessage = {
        role: 'assistant',
        content: response.data.response || response.data.error,
        error: response.data.error ? true : false,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: `Error: ${error.response?.data?.detail || error.message || 'Failed to get response'}`,
        error: true,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
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
              placeholder="Ask something about GeneXus..."
              className="chat-input"
              disabled={loading}
              data-testid="chat-input"
            />
            <button 
              type="submit" 
              className="send-button" 
              disabled={loading || !input.trim()}
              data-testid="send-button"
            >
              {loading ? '⏳' : '🚀'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
