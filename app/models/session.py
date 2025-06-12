from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime = datetime.now()

class UserSession(BaseModel):
    session_id: str
    user_id: str
    created_at: datetime = datetime.now()
    last_activity: datetime = datetime.now()
    chat_history: List[ChatMessage] = []
    context: Dict[str, Any] = {}
    
    def add_message(self, role: str, content: str):
        self.chat_history.append(ChatMessage(role=role, content=content))
        self.last_activity = datetime.now()
    
    def get_recent_messages(self, limit: int = 10) -> List[ChatMessage]:
        return self.chat_history[-limit:]
    
    def clear_history(self):
        self.chat_history = []
        self.last_activity = datetime.now()
    
    def update_context(self, key: str, value: Any):
        self.context[key] = value
        self.last_activity = datetime.now() 