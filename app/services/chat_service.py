from typing import List, Optional
from sqlalchemy.orm import Session
from app.models.chat import ChatSession, ChatMessage
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, db: Session):
        self.db = db

    def get_session(self, session_id: str, user_id: str) -> Optional[ChatSession]:
        """Get a chat session by ID and user ID"""
        try:
            return self.db.query(ChatSession).filter(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id,
                ChatSession.is_archived == False
            ).first()
        except Exception as e:
            logger.error(f"Error getting session: {str(e)}")
            raise

    def get_user_sessions(self, user_id: str, limit: int = 10) -> List[ChatSession]:
        """Get recent chat sessions for a user"""
        try:
            return self.db.query(ChatSession).filter(
                ChatSession.user_id == user_id,
                ChatSession.is_archived == False
            ).order_by(ChatSession.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"Error getting user sessions: {str(e)}")
            raise

    def get_session_messages(self, session_id: str, user_id: str, limit: int = 50) -> List[ChatMessage]:
        """Get messages from a chat session"""
        try:
            # Verify session belongs to user
            session = self.get_session(session_id, user_id)
            if not session:
                raise ValueError("Session not found or does not belong to user")

            return self.db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).order_by(ChatMessage.created_at.asc()).limit(limit).all()
        except Exception as e:
            logger.error(f"Error getting session messages: {str(e)}")
            raise

    def create_session(self, user_id: str) -> ChatSession:
        """Create a new chat session"""
        try:
            session = ChatSession(user_id=user_id)
            self.db.add(session)
            self.db.commit()
            self.db.refresh(session)
            return session
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating session: {str(e)}")
            raise

    def add_message(self, session_id: str, user_id: str, role: str, content: str) -> ChatMessage:
        """Add a message to a chat session"""
        try:
            # Verify session belongs to user
            session = self.get_session(session_id, user_id)
            if not session:
                raise ValueError("Session not found or does not belong to user")

            message = ChatMessage(
                session_id=session_id,
                role=role,
                content=content
            )
            self.db.add(message)
            self.db.commit()
            self.db.refresh(message)
            return message
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error adding message: {str(e)}")
            raise

    def archive_session(self, session_id: str, user_id: str) -> None:
        """Archive a chat session"""
        try:
            session = self.get_session(session_id, user_id)
            if not session:
                raise ValueError("Session not found or does not belong to user")

            session.is_archived = True
            session.archived_at = datetime.utcnow()
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error archiving session: {str(e)}")
            raise 