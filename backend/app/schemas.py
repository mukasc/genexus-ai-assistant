from typing import Optional
from pydantic import BaseModel

class ChatRequest(BaseModel): 
    message: str

class ChatResponse(BaseModel): 
    response: str
    context_used: bool
    error: Optional[str] = None
    retry_after: Optional[int] = None

class FeedbackRequest(BaseModel):
    user_question: str
    bot_response: str
    score: int # 1 para Like, -1 para Dislike
    comment: Optional[str] = None

class FeedbackResponse(BaseModel):
    status: str
    message: str

class IngestionResponse(BaseModel): 
    status: str
    message: str
    chunks_created: Optional[int] = None

class HealthResponse(BaseModel): 
    status: str
    api_key_configured: bool
    database_loaded: bool
    message: str
    app_name: str

class IndexStatusResponse(BaseModel): 
    exists: bool
    document_count: int
    collection_name: str
    message: str