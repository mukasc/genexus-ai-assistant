def test_root_endpoint(client):
    """Verifica se a API responde na raiz."""
    response = client.get("/api/")
    assert response.status_code == 200
    assert "version" in response.json()

def test_config_endpoint(client):
    """Verifica se o endpoint de configuração retorna o JSON correto."""
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "app_name" in data
    assert "primary_color" in data

def test_health_check(client):
    """Verifica se o health check retorna status (mesmo que degraded)."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    # Não checamos se é 'healthy' forçadamente, pois depende da API Key,
    # mas checamos se a estrutura da resposta está correta.
    assert "api_key_configured" in data