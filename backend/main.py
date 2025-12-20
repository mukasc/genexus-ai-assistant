import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import APP_CONFIG
from app.logging_config import logger
from app.core import rag
from app.api import routes

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
    lifespan=lifespan
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

@app.get("/api/")
async def root():
    return {"message": "White Label API", "version": "3.0.0 (Modular)"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)