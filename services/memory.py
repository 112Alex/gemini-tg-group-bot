from collections import deque
from typing import Dict, List, Any
from pydantic import BaseModel

class Message(BaseModel):
    """
    Representation of a single message in the conversation history.
    """
    role: str
    content: str


class ChatMemory:
    """
    In-memory storage for chat history with a fixed size per chat.
    """

    def __init__(self, max_history: int = 10) -> None:
        self._storage: Dict[int, deque[Message]] = {}
        self._max_history = max_history

    def add_message(self, chat_id: int, role: str, content: str) -> None:
        """
        Adds a message to the chat history.
        """
        if chat_id not in self._storage:
            self._storage[chat_id] = deque(maxlen=self._max_history)
        
        self._storage[chat_id].append(Message(role=role, content=content))

    def get_history(self, chat_id: int) -> List[Dict[str, str]]:
        """
        Returns the chat history in a format compatible with LLM providers.
        """
        if chat_id not in self._storage:
            return []
        
        return [msg.model_dump() for msg in self._storage[chat_id]]

    def clear_history(self, chat_id: int) -> None:
        """
        Clears the history for a specific chat.
        """
        if chat_id in self._storage:
            self._storage[chat_id].clear()


# Global instance for memory management
chat_memory = ChatMemory()
