import shutil
import os
from fastapi import APIRouter, HTTPException, Security
from app.core import rag
from app.config import APP_CONFIG, ROOT_DIR
from app.logging_config import logger
# from app.api.deps import verify_admin_access # <--- Segurança desativada para dev

router = APIRouter()

@router.get("/documents")
async def list_documents():
    """Lista todos os documentos (sources) indexados no banco."""
    try:
        if not rag.vectorstore_instance:
            rag.initialize_rag_system()
        
        vs = rag.vectorstore_instance
        if not vs: return {"count": 0, "documents": []}

        if hasattr(vs, '_collection'):
            data = vs._collection.get(include=['metadatas'])
            metadatas = data.get('metadatas', [])
            unique_sources = set()
            for meta in metadatas:
                if meta and 'source' in meta:
                    filename = os.path.basename(meta['source'])
                    unique_sources.add(filename)
            return {"count": len(unique_sources), "documents": sorted(list(unique_sources))}
        
        return {"count": 0, "documents": [], "message": "Store not supported"}
    except Exception as e:
        logger.error(f"Erro admin list: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents/{filename}/preview")
async def preview_document(filename: str):
    """Retorna uma amostra (primeiro chunk) do texto do documento."""
    try:
        if not rag.vectorstore_instance:
            rag.initialize_rag_system()
        
        vs = rag.vectorstore_instance
        if not hasattr(vs, '_collection'):
            raise HTTPException(status_code=400, detail="Database does not support preview")

        # 1. Busca metadados para achar o ID de um chunk deste arquivo
        # Nota: Isso pode ser lento se o banco for gigante, mas para uso local é ok.
        result = vs._collection.get(include=['metadatas'])
        target_id = None
        
        for i, meta in enumerate(result['metadatas']):
            if meta and 'source' in meta:
                if os.path.basename(meta['source']) == filename:
                    target_id = result['ids'][i]
                    break # Encontrou o primeiro pedaço, já serve
        
        # 2. Se achou, busca o conteúdo textual desse ID específico
        if target_id:
            chunk_data = vs._collection.get(ids=[target_id], include=['documents'])
            if chunk_data['documents']:
                content = chunk_data['documents'][0]
                # Retorna os primeiros 1000 caracteres
                return {"status": "success", "filename": filename, "preview": content[:1000]}
            else:
                return {"status": "warning", "message": "Chunk found but empty content."}
        else:
            raise HTTPException(status_code=404, detail="Document not found in index")

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erro admin preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Apaga um documento específico pelo nome."""
    try:
        if not rag.vectorstore_instance: rag.initialize_rag_system()
        vs = rag.vectorstore_instance
        if not hasattr(vs, '_collection'): raise HTTPException(status_code=400, detail="Not supported")

        result = vs._collection.get(include=['metadatas'])
        ids_to_delete = []
        for i, meta in enumerate(result['metadatas']):
            if meta and 'source' in meta:
                if os.path.basename(meta['source']) == filename:
                    ids_to_delete.append(result['ids'][i])
        
        if ids_to_delete:
            vs._collection.delete(ids=ids_to_delete)
            logger.info(f"Documento deletado: {filename}")
            return {"status": "success", "message": f"Deleted {filename}"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Erro admin delete: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/reset")
async def reset_database():
    try:
        db_path = APP_CONFIG.get('storage', {}).get('persist_directory', 'data/chroma_db')
        if not os.path.isabs(db_path): db_path = os.path.join(ROOT_DIR, db_path)
        rag.vectorstore_instance = None; rag.rag_chain = None
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            rag.initialize_rag_system()
            return {"status": "success", "message": "Database reset."}
        return {"status": "warning", "message": "Nothing to delete."}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))