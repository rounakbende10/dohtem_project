import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "open-ai-key")
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Validate required settings
    def __post_init__(self):
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")

settings = Settings()
