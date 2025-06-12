from pydantic_settings import BaseSettings
from typing import Optional
import secrets
import os

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Hybrid RAG System"
    
    # OpenAI Settings
    OPENAI_API_KEY: str="open-ai-key"
    
    # Vector Store Settings
    CHROMA_PERSIST_DIRECTORY: str = "chroma_db/"
    
    # Document Processing Settings
    MAX_CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Security Settings
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: list = ["*"]
    
    # Database Settings
    DATABASE_URL: str = "sqlite:///./data/app.db"  # Default to SQLite
    DB_TYPE: str = "sqlite"  # or "postgresql"
    
    # Create data directory if it doesn't exist
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.makedirs("data", exist_ok=True)
    
    class Config:
        env_file = ".env"

settings = Settings()
