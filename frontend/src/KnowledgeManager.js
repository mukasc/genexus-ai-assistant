import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css'; // Usa estilos globais ou crie específico

const KnowledgeManager = ({ onClose, backendUrl }) => {
    const [documents, setDocuments] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [deleting, setDeleting] = useState(null); // Guarda qual arquivo está sendo deletado

    const fetchDocuments = async () => {
        setLoading(true);
        try {
            const res = await axios.get(`${backendUrl}/api/admin/documents`);
            setDocuments(res.data.documents || []);
            setError(null);
        } catch (err) {
            console.error("Erro ao listar docs:", err);
            setError("Failed to load documents.");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchDocuments();
    }, []);

    const handleDelete = async (filename) => {
        if (!window.confirm(`Are you sure you want to remove "${filename}" from the Knowledge Base?`)) return;
        
        setDeleting(filename);
        try {
            await axios.delete(`${backendUrl}/api/admin/documents/${encodeURIComponent(filename)}`);
            // Remove da lista localmente para ser rápido
            setDocuments(prev => prev.filter(doc => doc !== filename));
        } catch (err) {
            alert(`Error deleting file: ${err.message}`);
        } finally {
            setDeleting(null);
        }
    };

    const handleReset = async () => {
        if (!window.confirm("DANGER: This will delete ALL documents. Are you sure?")) return;
        try {
            await axios.delete(`${backendUrl}/api/admin/reset`);
            setDocuments([]);
            alert("Database reset successfully.");
        } catch (err) {
            alert("Failed to reset database.");
        }
    };

    return (
        <div className="modal-overlay">
            <div className="modal-content">
                <div className="modal-header">
                    <h3>🗂️ Knowledge Manager</h3>
                    <button onClick={onClose} className="close-btn">×</button>
                </div>
                
                <div className="modal-body">
                    {loading ? (
                        <div className="loading-text">Loading index...</div>
                    ) : error ? (
                        <div className="error-text">{error}</div>
                    ) : documents.length === 0 ? (
                        <div className="empty-state">
                            <p>No documents found in the vector database.</p>
                            <small>Use "Add Knowledge" to upload PDFs.</small>
                        </div>
                    ) : (
                        <ul className="doc-list">
                            {documents.map((doc, idx) => (
                                <li key={idx} className="doc-item">
                                    <span className="doc-icon">📄</span>
                                    <span className="doc-name" title={doc}>{doc}</span>
                                    <button 
                                        onClick={() => handleDelete(doc)} 
                                        className="delete-btn"
                                        disabled={deleting === doc}
                                    >
                                        {deleting === doc ? '...' : '🗑️'}
                                    </button>
                                </li>
                            ))}
                        </ul>
                    )}
                </div>

                <div className="modal-footer">
                    <div className="stats">Total: {documents.length} files</div>
                    <button onClick={handleReset} className="danger-btn">Reset All</button>
                </div>
            </div>
        </div>
    );
};

export default KnowledgeManager;