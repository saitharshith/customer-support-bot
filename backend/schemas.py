from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

#Pydantic schemas for FASTAPI request/responce 

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Technical support question from the user.")
    thread_id: Optional[str] = Field(default="default_thread", description="Session thread ID for tracking conversation state.")

class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]
    rerank_score: float

class QueryResponse(BaseModel):
    question: str
    generation: str
    escalated: bool
    max_confidence_score: float
    context: List[DocumentChunk] = []