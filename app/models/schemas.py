from pydantic import BaseModel
from typing import List, Optional
from fastapi import UploadFile

class DocumentUpload(BaseModel):
    file_name: str
    content_type: str

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    use_hybrid: Optional[bool] = True

class RetrievedDocument(BaseModel):
    content: str
    metadata: dict
    score: float
    source: str

class QueryResponse(BaseModel):
    answer: str
    retrieved_documents: List[RetrievedDocument]
    query: str
    processing_time: float

class DocumentProcessResponse(BaseModel):
    message: str
    document_count: int
    chunks_created: int