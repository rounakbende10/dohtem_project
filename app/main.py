from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routers import rag
from app.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up RAG service...")
    print(f"ChromaDB directory: {settings.CHROMA_PERSIST_DIRECTORY}")
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
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(rag.router)

@app.get("/")
async def root():
    return {
        "message": "Hybrid RAG System API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/rag/upload",
            "query": "/rag/query",
            "stats": "/rag/stats",
            "health": "/rag/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)