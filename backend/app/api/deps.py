import os
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

# Nome do Header que o cliente deve enviar
API_KEY_NAME = "x-admin-key"

# Define o esquema de segurança
# auto_error=False permite que a gente trate o erro manualmente na função abaixo
api_key_scheme = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_admin_key():
    """Recupera a chave mestra do ambiente"""
    return os.getenv("ADMIN_SECRET_KEY", "genexus_admin_123")

async def verify_admin_access(key: str = Security(api_key_scheme)):
    """Dependência para proteger rotas sensíveis"""
    if not key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="❌ Acesso Negado: Header 'x-admin-key' ausente."
        )

    correct_key = get_admin_key()
    
    if key == correct_key:
        return True
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="❌ Acesso Negado: Chave de Administrador inválida."
    )