import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import APP_CONFIG
from app.logging_config import logger
from app.core import rag
from app.api import routes
from app.api import admin # <--- 1. NOVO IMPORT

# Ciclo de Vida (Inicialização do RAG)
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Startup: {APP_CONFIG.get('identity', {}).get('app_name')}")
    rag.initialize_rag_system()
    yield
    logger.info("Shutdown")

# Criação da App
app = FastAPI(
    title=APP_CONFIG.get('identity', {}).get('app_name'), 
    lifespan=lifespan,
    root_path="/proxy/8001" # <--- FIX: Ajuste para o Swagger funcionar no ambiente de Proxy/Preview
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusão de Rotas
app.include_router(routes.router, prefix="/api")
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"]) # <--- 2. NOVA ROTA

@app.get("/api/")
async def root():
    return {"message": "White Label API", "version": "3.1.0 (Admin Enabled)"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)