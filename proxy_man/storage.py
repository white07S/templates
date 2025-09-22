"""Storage layer implementation using TinyDB."""

from tinydb import TinyDB, Query, where
from tinydb.table import Document
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from models import ConversationDocument, Memory, Message
from config import settings

class TinyDBStore:
    """Base class for TinyDB storage operations."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db = TinyDB(db_path, sort_keys=True, indent=2)
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def insert_async(self, document: Dict[str, Any]) -> int:
        """Async insert operation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.db.insert, document)

    async def search_async(self, query: Query) -> List[Document]:
        """Async search operation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.db.search, query)

    async def update_async(self, fields: Dict[str, Any], query: Query) -> List[int]:
        """Async update operation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.db.update, fields, query)

    def close(self):
        """Close the database connection."""
        self.db.close()
        self.executor.shutdown(wait=True)

class ConversationStore(TinyDBStore):
    """Storage for conversation documents."""

    def __init__(self):
        super().__init__(settings.tinydb_conversations_path)
        self.conversations_table = self.db.table('conversations')

    async def save_conversation(self, conversation: ConversationDocument) -> int:
        """Save or update a conversation document."""
        doc = conversation.to_dict()

        # Check if conversation exists
        ConvQuery = Query()
        existing = self.conversations_table.search(
            (ConvQuery.session_id == conversation.session_id) &
            (ConvQuery.user_id == conversation.user_id)
        )

        if existing:
            # Update existing conversation
            return self.conversations_table.update(
                doc,
                (ConvQuery.session_id == conversation.session_id) &
                (ConvQuery.user_id == conversation.user_id)
            )[0]
        else:
            # Insert new conversation
            return self.conversations_table.insert(doc)

    async def get_conversation(self, session_id: str, user_id: str) -> Optional[ConversationDocument]:
        """Retrieve a conversation by session_id and user_id."""
        ConvQuery = Query()
        results = self.conversations_table.search(
            (ConvQuery.session_id == session_id) &
            (ConvQuery.user_id == user_id)
        )

        if results:
            data = results[0]
            # Convert timestamp strings back to datetime
            if isinstance(data.get('timestamp'), str):
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            for msg in data.get('messages', []):
                if isinstance(msg.get('timestamp'), str):
                    msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])
            return ConversationDocument(**data)
        return None

    async def get_user_conversations(self, user_id: str, limit: int = 10) -> List[ConversationDocument]:
        """Get recent conversations for a user."""
        ConvQuery = Query()
        results = self.conversations_table.search(ConvQuery.user_id == user_id)

        # Sort by timestamp and limit
        results.sort(key=lambda x: x['timestamp'], reverse=True)
        results = results[:limit]

        conversations = []
        for data in results:
            if isinstance(data.get('timestamp'), str):
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            for msg in data.get('messages', []):
                if isinstance(msg.get('timestamp'), str):
                    msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])
            conversations.append(ConversationDocument(**data))

        return conversations

    async def append_message(self, session_id: str, user_id: str, message: Message) -> bool:
        """Append a message to an existing conversation."""
        conversation = await self.get_conversation(session_id, user_id)

        if conversation:
            conversation.messages.append(message)
            await self.save_conversation(conversation)
            return True
        else:
            # Create new conversation with the message
            conversation = ConversationDocument(
                session_id=session_id,
                user_id=user_id,
                messages=[message]
            )
            await self.save_conversation(conversation)
            return True

    async def update_memory_state(self, session_id: str, user_id: str, memory_state: Dict[str, Any]) -> bool:
        """Update the memory state for a conversation."""
        ConvQuery = Query()
        result = self.conversations_table.update(
            {'memory_state': memory_state},
            (ConvQuery.session_id == session_id) &
            (ConvQuery.user_id == user_id)
        )
        return len(result) > 0

class MemoryStore(TinyDBStore):
    """Storage for memory documents."""

    def __init__(self):
        super().__init__(settings.tinydb_memories_path)
        self.memories_table = self.db.table('memories')

    async def save_memory(self, memory: Memory) -> str:
        """Save a memory document."""
        doc = memory.to_dict()
        doc_id = self.memories_table.insert(doc)
        return memory.memory_id

    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID."""
        MemQuery = Query()
        results = self.memories_table.search(MemQuery.memory_id == memory_id)

        if results:
            data = results[0]
            # Convert timestamp strings back to datetime
            if isinstance(data.get('created_at'), str):
                data['created_at'] = datetime.fromisoformat(data['created_at'])
            if isinstance(data.get('last_accessed'), str):
                data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
            if isinstance(data.get('last_modified'), str):
                data['last_modified'] = datetime.fromisoformat(data['last_modified'])
            return Memory(**data)
        return None

    async def get_user_memories(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Memory]:
        """Get memories for a user."""
        MemQuery = Query()

        if memory_type:
            results = self.memories_table.search(
                (MemQuery.user_id == user_id) &
                (MemQuery.memory_type == memory_type)
            )
        else:
            results = self.memories_table.search(MemQuery.user_id == user_id)

        # Sort by importance score and recency
        results.sort(key=lambda x: (x['importance_score'], x['last_accessed']), reverse=True)
        results = results[:limit]

        memories = []
        for data in results:
            if isinstance(data.get('created_at'), str):
                data['created_at'] = datetime.fromisoformat(data['created_at'])
            if isinstance(data.get('last_accessed'), str):
                data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
            if isinstance(data.get('last_modified'), str):
                data['last_modified'] = datetime.fromisoformat(data['last_modified'])
            memories.append(Memory(**data))

        return memories

    async def update_memory_access(self, memory_id: str) -> bool:
        """Update last accessed time and increment access frequency."""
        MemQuery = Query()
        memory = await self.get_memory(memory_id)

        if memory:
            result = self.memories_table.update(
                {
                    'last_accessed': datetime.utcnow().isoformat(),
                    'access_frequency': memory.access_frequency + 1
                },
                MemQuery.memory_id == memory_id
            )
            return len(result) > 0
        return False

    async def update_memory_content(
        self,
        memory_id: str,
        content: str,
        compressed_content: Optional[str] = None
    ) -> bool:
        """Update memory content."""
        MemQuery = Query()
        updates = {
            'content': content,
            'last_modified': datetime.utcnow().isoformat()
        }

        if compressed_content:
            updates['compressed_content'] = compressed_content

        result = self.memories_table.update(
            updates,
            MemQuery.memory_id == memory_id
        )
        return len(result) > 0

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        MemQuery = Query()
        result = self.memories_table.remove(MemQuery.memory_id == memory_id)
        return len(result) > 0

    async def search_memories_by_keywords(
        self,
        user_id: str,
        keywords: List[str],
        limit: int = 20
    ) -> List[Memory]:
        """Search memories by keywords."""
        MemQuery = Query()
        results = self.memories_table.search(
            (MemQuery.user_id == user_id) &
            (MemQuery.keywords.any(keywords))
        )

        results = results[:limit]

        memories = []
        for data in results:
            if isinstance(data.get('created_at'), str):
                data['created_at'] = datetime.fromisoformat(data['created_at'])
            if isinstance(data.get('last_accessed'), str):
                data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
            if isinstance(data.get('last_modified'), str):
                data['last_modified'] = datetime.fromisoformat(data['last_modified'])
            memories.append(Memory(**data))

        return memories