from pydantic import BaseModel
from typing import List, Optional
from fastapi import UploadFile
from datetime import datetime
from enum import Enum

class DocumentStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"

class DocumentMetadata(BaseModel):
    file_name: str
    content_type: str
    upload_date: datetime
    last_modified: datetime
    owner_id: str
    status: DocumentStatus = DocumentStatus.ACTIVE
    version: int = 1
    tags: List[str] = []
    description: Optional[str] = None

class DocumentUpload(BaseModel):
    file_name: str
    content_type: str
    owner_id: str
    tags: List[str] = []
    description: Optional[str] = None

class DocumentUpdate(BaseModel):
    status: Optional[DocumentStatus] = None
    tags: Optional[List[str]] = None
    description: Optional[str] = None

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