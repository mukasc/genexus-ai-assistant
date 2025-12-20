import os
import json
import time
from fastapi import APIRouter, HTTPException
from app.config import ROOT_DIR
from app.logging_config import logger
from app.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter()

FEEDBACK_FILE = os.path.join(ROOT_DIR, "data", "feedback_log.json")

@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """
    Salva o feedback do usuário (Like/Dislike) em um arquivo JSON local.
    """
    try:
        # Cria a estrutura do registro
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "score": feedback.score,
            "question": feedback.user_question,
            "response_preview": feedback.bot_response[:100] + "...",
            "comment": feedback.comment
        }

        # Loga no sistema (Vai para o Loki/Grafana automaticamente)
        log_extra = {
            "user_score": feedback.score,
            "user_comment": feedback.comment or "No comment"
        }
        if feedback.score > 0:
            logger.info("👍 Feedback Positivo Recebido", extra=log_extra)
        else:
            logger.warning("👎 Feedback Negativo Recebido", extra=log_extra)

        # Salva no arquivo JSON (Append)
        # Lê existente
        existing_data = []
        if os.path.exists(FEEDBACK_FILE):
            try:
                with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                pass # Arquivo vazio ou corrompido, começa do zero

        # Adiciona novo
        existing_data.append(entry)

        # Escreve de volta
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

        return FeedbackResponse(status="success", message="Feedback recorded")

    except Exception as e:
        logger.error(f"Erro ao salvar feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback")

@router.get("/summary")
async def get_feedback_summary():
    """Retorna um resumo simples dos feedbacks."""
    if not os.path.exists(FEEDBACK_FILE):
        return {"total": 0, "likes": 0, "dislikes": 0}
    
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        total = len(data)
        likes = sum(1 for item in data if item.get("score", 0) > 0)
        dislikes = sum(1 for item in data if item.get("score", 0) < 0)
        
        return {
            "total": total,
            "likes": likes,
            "dislikes": dislikes,
            "recent": data[-5:] # Retorna os 5 últimos
        }
    except:
        return {"error": "Could not read feedback file"}