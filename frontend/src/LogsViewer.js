import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './LogsViewer.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

const LogsViewer = ({ onClose }) => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [lines, setLines] = useState(100);
  const [levelFilter, setLevelFilter] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [error, setError] = useState(null);
  const intervalRef = useRef(null);
  const logsEndRef = useRef(null);

  const scrollToBottom = () => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const fetchLogs = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const params = new URLSearchParams();
      params.append('lines', lines);
      if (levelFilter !== 'all') {
        params.append('level', levelFilter);
      }
      if (searchTerm) {
        params.append('search', searchTerm);
      }

      const response = await axios.get(`${BACKEND_URL.replace('/api', '')}/api/logs?${params}`);
      
      if (response.data.error) {
        setError(response.data.error);
      } else {
        setLogs(response.data.logs || []);
      }
    } catch (err) {
      setError(`Erro ao buscar logs: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
  }, [lines, levelFilter, searchTerm]);

  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(() => {
        fetchLogs();
      }, 3000);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [autoRefresh, lines, levelFilter, searchTerm]);

  useEffect(() => {
    if (autoRefresh) {
      scrollToBottom();
    }
  }, [logs, autoRefresh]);

  const getLevelColor = (level) => {
    switch (level?.toLowerCase()) {
      case 'error':
        return 'log-error';
      case 'warning':
        return 'log-warning';
      case 'info':
        return 'log-info';
      case 'debug':
        return 'log-debug';
      default:
        return '';
    }
  };

  const getLevelIcon = (level) => {
    switch (level?.toLowerCase()) {
      case 'error':
        return '❌';
      case 'warning':
        return '⚠️';
      case 'info':
        return 'ℹ️';
      case 'debug':
        return '🔍';
      default:
        return '📝';
    }
  };

  const downloadLogs = () => {
    const content = logs.map(log => JSON.stringify(log)).join('\n');
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `genexus-logs-${new Date().toISOString()}.log`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const copyToClipboard = (log) => {
    navigator.clipboard.writeText(JSON.stringify(log, null, 2));
  };

  return (
    <div className="logs-viewer-overlay">
      <div className="logs-viewer">
        {/* Header */}
        <div className="logs-header">
          <h2>📊 Application Logs</h2>
          <button className="close-button" onClick={onClose}>×</button>
        </div>

        {/* Controls */}
        <div className="logs-controls">
          <div className="control-group">
            <label>Linhas:</label>
            <select value={lines} onChange={(e) => setLines(Number(e.target.value))}>
              <option value={50}>50</option>
              <option value={100}>100</option>
              <option value={200}>200</option>
              <option value={500}>500</option>
            </select>
          </div>

          <div className="control-group">
            <label>Level:</label>
            <select value={levelFilter} onChange={(e) => setLevelFilter(e.target.value)}>
              <option value="all">Todos</option>
              <option value="error">Errors</option>
              <option value="warning">Warnings</option>
              <option value="info">Info</option>
              <option value="debug">Debug</option>
            </select>
          </div>

          <div className="control-group search-group">
            <label>🔍</label>
            <input
              type="text"
              placeholder="Buscar..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
          </div>

          <div className="control-group">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
              Auto-refresh (3s)
            </label>
          </div>

          <button onClick={fetchLogs} className="refresh-button" disabled={loading}>
            🔄 Atualizar
          </button>

          <button onClick={downloadLogs} className="download-button" disabled={logs.length === 0}>
            💾 Download
          </button>
        </div>

        {/* Error Message */}
        {error && (
          <div className="logs-error">
            ⚠️ {error}
          </div>
        )}

        {/* Stats */}
        <div className="logs-stats">
          <span>Total: {logs.length}</span>
          <span>Errors: {logs.filter(l => l.level === 'error').length}</span>
          <span>Warnings: {logs.filter(l => l.level === 'warning').length}</span>
          {loading && <span className="loading-indicator">⏳ Carregando...</span>}
        </div>

        {/* Logs Content */}
        <div className="logs-content">
          {logs.length === 0 ? (
            <div className="no-logs">
              {loading ? '⏳ Carregando logs...' : '📭 Nenhum log encontrado'}
            </div>
          ) : (
            logs.map((log, index) => (
              <div
                key={index}
                className={`log-entry ${getLevelColor(log.level)}`}
              >
                <div className="log-header-line">
                  <span className="log-icon">{getLevelIcon(log.level)}</span>
                  <span className="log-level">{log.level || 'info'}</span>
                  <span className="log-name">{log.name || 'unknown'}</span>
                  <span className="log-function">{log.funcName}</span>
                  <span className="log-line">:{log.lineno}</span>
                  <button
                    className="copy-button"
                    onClick={() => copyToClipboard(log)}
                    title="Copiar log"
                  >
                    📋
                  </button>
                </div>
                <div className="log-message">{log.message}</div>
                {log.module && (
                  <div className="log-meta">
                    <span className="meta-item">📦 {log.module}</span>
                  </div>
                )}
                {/* Extra fields */}
                {Object.keys(log).filter(key => 
                  !['timestamp', 'level', 'name', 'message', 'module', 'funcName', 'lineno'].includes(key)
                ).length > 0 && (
                  <div className="log-extra">
                    {Object.entries(log)
                      .filter(([key]) => !['timestamp', 'level', 'name', 'message', 'module', 'funcName', 'lineno'].includes(key))
                      .map(([key, value]) => (
                        <span key={key} className="extra-field">
                          <strong>{key}:</strong> {JSON.stringify(value)}
                        </span>
                      ))}
                  </div>
                )}
              </div>
            ))
          )}
          <div ref={logsEndRef} />
        </div>
      </div>
    </div>
  );
};

export default LogsViewer;
