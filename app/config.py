from pydantic_settings import BaseSettings
from typing import Optional, List
import secrets
import os
import dotenv

dotenv.load_dotenv()

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Hybrid RAG System"
    
    # OpenAI Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL")
    OPENAI_TEMPERATURE: float = 0.1
    #OPENAI_MAX_TOKENS: int = 500
    
    # Vector Store Settings
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY")
    CHROMA_COLLECTION_NAME: str = "documents"
    CHROMA_SPACE: str = "cosine"
    
    # Document Processing Settings
    MAX_CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    ALLOWED_FILE_TYPES: List[str] = [".pdf", ".docx"]
    
    # Security Settings
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Database Settings
    DATABASE_URL: str = "sqlite:///./data/app.db"
    DB_TYPE: str = "sqlite"
    
    # Create data directory if it doesn't exist
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.makedirs("data", exist_ok=True)
    
    class Config:
        env_file = ".env"

settings = Settings()
