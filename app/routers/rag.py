from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List, Dict, Any
import asyncio

from app.services.rag_service import RAGService
from app.models.schemas import QueryRequest, QueryResponse, DocumentProcessResponse

router = APIRouter(prefix="/rag", tags=["RAG"])

# Global RAG service instance
rag_service = RAGService()

async def get_rag_service() -> RAGService:
    return rag_service

@router.post("/upload", response_model=DocumentProcessResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    service: RAGService = Depends(get_rag_service)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    total_chunks = 0
    processed_files = 0
    
    for file in files:
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.docx')):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.filename}. Only PDF and DOCX files are supported."
            )
        
        try:
            # Read file content
            content = await file.read()
            
            # Process document
            result = await service.process_documents(content, file.filename)
            total_chunks += result["chunks_created"]
            processed_files += 1
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing file {file.filename}: {str(e)}"
            )
    
    return DocumentProcessResponse(
        message=f"Successfully processed {processed_files} file(s)",
        document_count=processed_files,
        chunks_created=total_chunks
    )

@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    service: RAGService = Depends(get_rag_service)
):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        response = await service.query(
            query=request.query,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid
        )
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@router.get("/stats")
async def get_stats(service: RAGService = Depends(get_rag_service)):
    return await service.get_stats()

@router.delete("/clear")
async def clear_database(service: RAGService = Depends(get_rag_service)):
    return await service.clear_database()

@router.get("/health")
async def health_check():
    return {"status": "healthy", "message": "RAG service is running"}