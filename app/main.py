from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from app.db.database import get_db, init_db
from app.services.chat_service import ChatService
from app.services.rag_service import RAGService
from app.models.chat import ChatMessage, ChatSession
from app.config import settings
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up RAG service...")
    print(f"ChromaDB directory: {settings.CHROMA_PERSIST_DIRECTORY}")
    init_db()
    logger.info("Database initialized successfully")
    yield
    # Shutdown
    print("Shutting down RAG service...")

app = FastAPI(
    title="Hybrid RAG System",
    description="A hybrid RAG system using ChromaDB, BM25, and OpenAI embeddings",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # To be Configured for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class CreateSessionRequest(BaseModel):
    user_id: str

class ChatRequest(BaseModel):
    message: str
    user_id: str

class QueryRequest(BaseModel):
    query: str
    session_id: str
    user_id: str

# Initialize services
chat_service = ChatService(next(get_db()))
rag_service = RAGService()

@app.get("/")
async def root():
    return {
        "message": "Hybrid RAG System API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload",
            "query": "/query",
            "sessions": "/sessions",
            "chat": "/chat/{session_id}",
            "docs": "/docs"
        }
    }

@app.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Upload documents for processing"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        results = []
        for file in files:
            # Validate file type
            if not file.filename.endswith(('.pdf', '.docx')):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is not supported. Only PDF and DOCX files are allowed."
                )

            # Process the file
            result = await rag_service.process_file(file)
            results.append({
                "filename": file.filename,
                "status": result["status"],
                "message": result["message"],
                "chunks": result["chunks"]
            })

        return {
            "message": "Documents processed successfully",
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """Query the RAG system"""
    try:
        response = await rag_service.query(
            query=request.query,
            session_id=request.session_id,
            user_id=request.user_id
        )
        return {"response": response}
    except Exception as e:
        logger.error(f"Error querying documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{user_id}")
async def get_user_sessions(
    user_id: str,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get recent chat sessions for a user"""
    try:
        sessions = chat_service.get_user_sessions(user_id, limit)
        return [{
            "session_id": session.id,
            "user_id": session.user_id,
            "created_at": session.created_at,
            "is_archived": session.is_archived
        } for session in sessions]
    except Exception as e:
        logger.error(f"Error getting user sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    user_id: str,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get messages from a chat session"""
    try:
        messages = chat_service.get_session_messages(session_id, user_id, limit)
        return [{
            "role": message.role,
            "content": message.content,
            "created_at": message.created_at
        } for message in messages]
    except Exception as e:
        logger.error(f"Error getting session messages: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions")
async def create_chat_session(
    request: CreateSessionRequest,
    db: Session = Depends(get_db)
):
    """Create a new chat session"""
    try:
        if not request.user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
            
        session = chat_service.create_session(request.user_id)
        return {
            "session_id": session.id,
            "user_id": session.user_id,
            "created_at": session.created_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating chat session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@app.delete("/sessions/{session_id}")
async def archive_session(
    session_id: str,
    user_id: str,
    db: Session = Depends(get_db)
):
    """Archive a chat session"""
    try:
        chat_service.archive_session(session_id, user_id)
        return {"message": "Session archived successfully"}
    except Exception as e:
        logger.error(f"Error archiving session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/{session_id}")
async def chat(
    session_id: str,
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """Send a message in a chat session"""
    try:
        # Verify session exists and belongs to user
        session = chat_service.get_session(session_id, request.user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or does not belong to user")
        
        # Add user message to chat history
        user_message = chat_service.add_message(
            session_id=session_id,
            user_id=request.user_id,
            role="user",
            content=request.message
        )
        
        # Get chat history for context
        chat_history = chat_service.get_session_messages(session_id, request.user_id)
        
        # Process through RAG system
        response = await rag_service.query(
            query=request.message,
            session_id=session_id,
            user_id=request.user_id
        )
        
        # Add assistant response to chat history
        assistant_message = chat_service.add_message(
            session_id=session_id,
            user_id=request.user_id,
            role="assistant",
            content=response
        )
        
        return {
            "message": response,
            "session_id": session_id,
            "user_id": request.user_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)