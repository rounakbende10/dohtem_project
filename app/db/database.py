from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings
import os
import logging
from sqlalchemy import inspect

logger = logging.getLogger(__name__)

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Configure SQLAlchemy engine based on database type
if settings.DB_TYPE == "sqlite":
    # SQLite configuration
    SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False}  # Needed for SQLite
    )
else:
    # PostgreSQL configuration
    SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class
Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize database
def init_db():
    """Initialize the database and create all tables"""
    from app.models.chat import ChatSession, ChatMessage  # Import models here to avoid circular imports
    
    logger.info("Dropping all existing tables...")
    Base.metadata.drop_all(bind=engine)
    
    logger.info("Creating new tables...")
    Base.metadata.create_all(bind=engine)
    
    # Verify tables were created
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    logger.info(f"Created tables: {tables}")
    
    # Log table schemas
    for table in tables:
        columns = inspector.get_columns(table)
        logger.info(f"Table {table} schema:")
        for column in columns:
            logger.info(f"  {column['name']}: {column['type']}") 