from unittest.mock import patch, MagicMock

def test_chat_flow_mocked(client):
    """
    Testa o fluxo completo do endpoint /api/chat SIMULANDO a IA.
    Isso garante que a rota, os logs e o tratamento de resposta funcionam,
    sem gastar dinheiro ou depender da API do Google.
    """
    
    # 1. Dados simulados que o RAG retornaria
    mock_rag_response = {
        "response": "Esta é uma resposta simulada de teste.",
        "sources": [
            MagicMock(metadata={"source": "/data/uploads/manual_teste.pdf"})
        ]
    }

    # 2. Patching (Substituição temporária das funções reais)
    # Substituímos a função que executa a Chain e a verificação de inicialização
    with patch("app.api.routes.rag.run_chain_with_retry", return_value=mock_rag_response) as mock_run:
        with patch("app.api.routes.rag.rag_chain", MagicMock()): # Finge que o RAG está inicializado
            
            # 3. Executa a requisição real para a API
            payload = {"message": "Teste de conexão", "session_id": "sessao-teste-123"}
            response = client.post("/api/chat", json=payload)

            # 4. Verificações (Asserts)
            assert response.status_code == 200
            data = response.json()
            
            # Verifica se a resposta veio do nosso Mock
            assert "Esta é uma resposta simulada" in data["response"]
            # Verifica se a formatação de fontes aconteceu (lógica do routes.py)
            assert "manual_teste.pdf" in data["response"]
            assert data["context_used"] is True

def test_chat_validation_error(client):
    """Testa se a API rejeita mensagem vazia corretamente."""
    # Envia payload inválido (sem message)
    response = client.post("/api/chat", json={"session_id": "123"})
    assert response.status_code == 422 # Unprocessable Entity (Pydantic validation)