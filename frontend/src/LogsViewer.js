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
  
  // Filtros de Tempo
  const [timeRange, setTimeRange] = useState('15m'); 
  const [customStart, setCustomStart] = useState('');
  const [customEnd, setCustomEnd] = useState('');

  const intervalRef = useRef(null);
  const logsEndRef = useRef(null);

  const fetchLogs = async () => {
    if (!autoRefresh) setLoading(true);
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

      // Cálculo de Tempo
      if (timeRange !== 'all') {
        const now = new Date();
        let startDate = null;
        let endDate = null;

        if (timeRange === 'absolute') {
          if (customStart) startDate = new Date(customStart);
          if (customEnd) endDate = new Date(customEnd);
        } else {
          startDate = new Date(now.getTime());
          switch (timeRange) {
            case '1m': startDate.setMinutes(now.getMinutes() - 1); break;
            case '5m': startDate.setMinutes(now.getMinutes() - 5); break;
            case '10m': startDate.setMinutes(now.getMinutes() - 10); break;
            case '15m': startDate.setMinutes(now.getMinutes() - 15); break;
            case '30m': startDate.setMinutes(now.getMinutes() - 30); break;
            case '1h': startDate.setHours(now.getHours() - 1); break;
            case '24h': startDate.setHours(now.getHours() - 24); break;
            default: break;
          }
        }

        if (startDate) params.append('start_time', startDate.toISOString().split('.')[0]);
        if (endDate) params.append('end_time', endDate.toISOString().split('.')[0]);
      }

      const baseUrl = BACKEND_URL.includes('/api') ? BACKEND_URL : `${BACKEND_URL}/api`;
      const response = await axios.get(`${baseUrl}/logs?${params}`);
      
      if (response.data.error) {
        setError(response.data.error);
      } else {
        const receivedLogs = response.data.logs || [];
        setLogs(receivedLogs.reverse());
      }
    } catch (err) {
      if (!autoRefresh) setError(`Erro: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
  }, [lines, levelFilter, searchTerm, timeRange, customStart, customEnd]); 

  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(() => {
        fetchLogs();
      }, 3000);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => clearInterval(intervalRef.current);
  }, [autoRefresh, lines, levelFilter, searchTerm, timeRange]);

  const getLevelColor = (level) => {
    switch (level?.toLowerCase()) {
      case 'error': return 'log-error';
      case 'warning': return 'log-warning';
      case 'info': return 'log-info';
      case 'debug': return 'log-debug';
      default: return '';
    }
  };

  const getLevelIcon = (level) => {
    switch (level?.toLowerCase()) {
      case 'error': return '❌';
      case 'warning': return '⚠️';
      case 'info': return 'ℹ️';
      case 'debug': return '🔍';
      default: return '📝';
    }
  };

  const copyToClipboard = (log) => {
    navigator.clipboard.writeText(JSON.stringify(log, null, 2));
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

  return (
    <div className="logs-viewer-overlay">
      <div className="logs-viewer">
        <div className="logs-header">
          <h2>📊 Application Logs</h2>
          <button className="close-button" onClick={onClose}>×</button>
        </div>

        <div className="logs-controls">
          <div className="control-group">
            <label>Tempo:</label>
            <select value={timeRange} onChange={(e) => setTimeRange(e.target.value)} className="time-select">
              <option value="1m">Último 1 min</option>
              <option value="5m">Últimos 5 min</option>
              <option value="10m">Últimos 10 min</option>
              <option value="15m">Últimos 15 min</option>
              <option value="30m">Últimos 30 min</option>
              <option value="1h">Última 1 hora</option>
              <option value="24h">Últimas 24 horas</option>
              <option value="all">Todo o Histórico</option>
              <option value="absolute">Personalizado</option>
            </select>
          </div>

          {timeRange === 'absolute' && (
            <div className="control-group absolute-dates">
              <input 
                type="datetime-local" 
                value={customStart} 
                onChange={(e) => setCustomStart(e.target.value)} 
                className="date-input"
              />
              <span style={{color:'white'}}>-</span>
              <input 
                type="datetime-local" 
                value={customEnd} 
                onChange={(e) => setCustomEnd(e.target.value)} 
                className="date-input"
              />
            </div>
          )}

          <div className="control-group">
            <label>Linhas:</label>
            <select value={lines} onChange={(e) => setLines(Number(e.target.value))}>
              <option value={50}>50</option>
              <option value={100}>100</option>
              <option value={200}>200</option>
              <option value={500}>500</option>
              <option value={1000}>1000</option>
            </select>
          </div>

          <div className="control-group">
            <label>Nível:</label>
            <select value={levelFilter} onChange={(e) => setLevelFilter(e.target.value)}>
              <option value="all">Todos</option>
              <option value="error">Errors</option>
              <option value="warning">Warnings</option>
              <option value="info">Info</option>
              {/* --- OPÇÃO DEBUG REINSERIDA AQUI --- */}
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
              Auto (3s)
            </label>
          </div>

          <button onClick={fetchLogs} className="refresh-button" disabled={loading} title="Atualizar">
            🔄
          </button>
          
          <button onClick={downloadLogs} className="download-button" disabled={logs.length === 0} title="Baixar Logs">
            💾
          </button>
        </div>

        {error && <div className="logs-error">⚠️ {error}</div>}

        <div className="logs-stats">
          <span>Exibindo: {logs.length} logs (Mais recentes primeiro)</span>
          {timeRange !== 'all' && <span className="filter-badge">⏳ Filtro de Tempo Ativo</span>}
          <span>Errors: {logs.filter(l => l.level === 'ERROR').length}</span>
          <span>Warnings: {logs.filter(l => l.level === 'WARNING').length}</span>
          {loading && <span className="loading-indicator">⏳ Buscando...</span>}
        </div>

        <div className="logs-content">
          {logs.length === 0 ? (
            <div className="no-logs">
              {loading ? '⏳ Carregando logs...' : '📭 Nenhum log encontrado para este período/filtro.'}
            </div>
          ) : (
            logs.map((log, index) => (
              <div key={index} className={`log-entry ${getLevelColor(log.level)}`}>
                <div className="log-header-line">
                  <span className="log-icon">{getLevelIcon(log.level)}</span>
                  <span className="log-time">
                    {log.timestamp ? log.timestamp.replace('T', ' ') : 'No Date'}
                  </span>
                  <span className="log-level">{log.level || 'INFO'}</span>
                  
                  {(log.name || log.module) && (
                     <span className="log-name">[{log.module || log.name}]</span>
                  )}

                  <span className="log-function">{log.funcName ? ` ${log.funcName}()` : ''}</span>
                  <span className="log-line">{log.lineno ? `:${log.lineno}` : ''}</span>
                  
                  <button className="copy-button" onClick={() => copyToClipboard(log)} title="Copiar JSON">
                    📋
                  </button>
                </div>
                
                <div className="log-message">{log.message}</div>
                
                {/* Metadados Extras */}
                {Object.keys(log).filter(key => 
                  !['timestamp', 'level', 'name', 'message', 'module', 'funcName', 'lineno', 'asctime', 'taskName'].includes(key)
                ).length > 0 && (
                  <div className="log-extra">
                    {Object.entries(log)
                      .filter(([key]) => !['timestamp', 'level', 'name', 'message', 'module', 'funcName', 'lineno', 'asctime', 'taskName'].includes(key))
                      .map(([key, value]) => (
                        <span key={key} className="extra-field">
                          <strong>{key}:</strong> {typeof value === 'object' ? JSON.stringify(value) : String(value)}
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