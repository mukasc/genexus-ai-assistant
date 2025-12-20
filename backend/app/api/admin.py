import shutil
import os
from fastapi import APIRouter, HTTPException, Security # <--- Importe Security
from app.core import rag
from app.config import APP_CONFIG, ROOT_DIR
from app.logging_config import logger
from app.api.deps import verify_admin_access

# MUDANÇA AQUI: Usamos Security() dentro da lista de dependências
router = APIRouter(dependencies=[Security(verify_admin_access)])

@router.get("/documents")
async def list_documents():
    """(Protegido) Lista todos os documentos indexados."""
    try:
        if not rag.vectorstore_instance:
            rag.initialize_rag_system()
        
        vs = rag.vectorstore_instance
        if not vs:
            return {"count": 0, "documents": []}

        if hasattr(vs, '_collection'):
            data = vs._collection.get()
            metadatas = data.get('metadatas', [])
            unique_sources = set()
            for meta in metadatas:
                if meta and 'source' in meta:
                    filename = os.path.basename(meta['source'])
                    unique_sources.add(filename)
            return {"count": len(unique_sources), "documents": list(unique_sources)}
        
        return {"count": 0, "documents": [], "message": "Store not supported for listing"}

    except Exception as e:
        logger.error(f"Erro admin: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/reset")
async def reset_database():
    """(Protegido) DANGER: Apaga o banco de dados."""
    try:
        db_path = APP_CONFIG.get('storage', {}).get('persist_directory', 'data/chroma_db')
        if not os.path.isabs(db_path):
            db_path = os.path.join(ROOT_DIR, db_path)

        rag.vectorstore_instance = None
        rag.rag_chain = None
        
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            logger.warning(f"Banco apagado pelo Admin: {db_path}")
            rag.initialize_rag_system()
            return {"status": "success", "message": "Database reset successfully."}
        else:
            return {"status": "warning", "message": "Nothing to delete."}
            
    except Exception as e:
        logger.error(f"Erro reset: {e}")
        raise HTTPException(status_code=500, detail=str(e))