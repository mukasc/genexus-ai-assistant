import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const KnowledgeManager = ({ onClose, backendUrl }) => {
    const [documents, setDocuments] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [deleting, setDeleting] = useState(null);
    
    // States para o Preview
    const [previewDoc, setPreviewDoc] = useState(null); // Nome do doc sendo visualizado
    const [previewContent, setPreviewContent] = useState("");
    const [loadingPreview, setLoadingPreview] = useState(false);

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
        if (!window.confirm(`Remove "${filename}"?`)) return;
        setDeleting(filename);
        try {
            // Encode para suportar espaços no nome
            await axios.delete(`${backendUrl}/api/admin/documents/${encodeURIComponent(filename)}`);
            setDocuments(prev => prev.filter(doc => doc !== filename));
        } catch (err) {
            alert(`Error: ${err.message}`);
        } finally {
            setDeleting(null);
        }
    };

    const handlePreview = async (filename) => {
        setPreviewDoc(filename);
        setLoadingPreview(true);
        setPreviewContent("");
        
        try {
            const res = await axios.get(`${backendUrl}/api/admin/documents/${encodeURIComponent(filename)}/preview`);
            setPreviewContent(res.data.preview);
        } catch (err) {
            setPreviewContent("Error loading preview. File might be corrupted or missing.");
        } finally {
            setLoadingPreview(false);
        }
    };

    const handleReset = async () => {
        if (!window.confirm("DELETE ALL documents?")) return;
        try { await axios.delete(`${backendUrl}/api/admin/reset`); setDocuments([]); alert("Reset done."); } catch { alert("Failed."); }
    };

    return (
        <div className="modal-overlay">
            <div className="modal-content">
                <div className="modal-header">
                    <h3>🗂️ Knowledge Manager</h3>
                    <button onClick={onClose} className="close-btn">×</button>
                </div>
                
                <div className="modal-body">
                    
                    {/* Se estiver vendo preview, mostra o texto em vez da lista */}
                    {previewDoc ? (
                        <div className="preview-container">
                            <div className="preview-header">
                                <strong>📄 {previewDoc}</strong>
                                <button onClick={() => setPreviewDoc(null)} className="back-btn">⬅️ Back to list</button>
                            </div>
                            <div className="preview-text-box">
                                {loadingPreview ? "Loading text sample..." : previewContent}
                            </div>
                            <small className="preview-note">* Showing first 1000 characters.</small>
                        </div>
                    ) : (
                        // Lista Normal
                        <>
                            {loading ? <div className="loading-text">Loading index...</div> : 
                             error ? <div className="error-text">{error}</div> : 
                             documents.length === 0 ? (
                                <div className="empty-state"><p>No documents found.</p></div>
                            ) : (
                                <ul className="doc-list">
                                    {documents.map((doc, idx) => (
                                        <li key={idx} className="doc-item">
                                            <span className="doc-icon">📄</span>
                                            <span className="doc-name" title={doc}>{doc}</span>
                                            <div className="doc-actions">
                                                <button 
                                                    onClick={() => handlePreview(doc)} 
                                                    className="icon-btn preview-btn"
                                                    title="Preview Content"
                                                >
                                                    👁️
                                                </button>
                                                <button 
                                                    onClick={() => handleDelete(doc)} 
                                                    className="icon-btn delete-btn"
                                                    disabled={deleting === doc}
                                                    title="Delete"
                                                >
                                                    {deleting === doc ? '...' : '🗑️'}
                                                </button>
                                            </div>
                                        </li>
                                    ))}
                                </ul>
                            )}
                        </>
                    )}
                </div>

                <div className="modal-footer">
                    {!previewDoc && (
                        <>
                            <div className="stats">Total: {documents.length} files</div>
                            <button onClick={handleReset} className="danger-btn">Reset All</button>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
};

export default KnowledgeManager;