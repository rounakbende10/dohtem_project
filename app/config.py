from pydantic_settings import BaseSettings
from typing import Optional, List
import secrets
import os
import dotenv


class Settings(BaseSettings):
    # OpenAI Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL")
    OPENAI_TEMPERATURE: float = os.getenv("OPENAI_TEMPERATURE")
    #OPENAI_MAX_TOKENS: int = 500
    
    # Vector Store Settings
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME")
    CHROMA_SPACE: str = "cosine"
    
    # Document Processing Settings
    MAX_CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    ALLOWED_FILE_TYPES: List[str] = [".pdf", ".docx"]
    
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
