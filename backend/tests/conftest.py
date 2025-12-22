import pytest
import sys
import os
from fastapi.testclient import TestClient

# 1. Adiciona o diretório 'backend' ao PATH do Python
# Isso permite que os testes importem 'main' e 'app' como se estivessem na raiz
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

# 2. Fixture do Cliente
# Cria uma instância do cliente de teste que será injetada nas funções de teste
@pytest.fixture(scope="module")
def client():
    # O TestClient roda a aplicação FastAPI localmente sem precisar subir servidor
    with TestClient(app) as c:
        yield c