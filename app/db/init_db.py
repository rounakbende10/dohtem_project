from app.db.database import init_db, SessionLocal, Base
from app.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init():
    """Initialize the database"""
    logger.info("Creating initial database tables")
    init_db()
    logger.info("Database tables created successfully")

def main():
    """Main function to initialize the database"""
    logger.info("Initializing database")
    init()
    logger.info("Database initialization completed")

if __name__ == "__main__":
    main() 