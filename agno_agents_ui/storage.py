"""Storage layer implementation using TinyDB with performance optimizations."""

from tinydb import TinyDB, Query, where
from tinydb.table import Document
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from models import ConversationDocument, Memory, Message
from config import settings
from cache_utils import TTLCache, LRUCache, BatchProcessor, cache_key_from_args

class TinyDBStore:
    """Base class for TinyDB storage operations with caching and connection pooling."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db = TinyDB(db_path, sort_keys=True, indent=2)
        self.executor = ThreadPoolExecutor(max_workers=8)  # Increased for better concurrency

        # Performance optimizations
        self.query_cache = TTLCache(maxsize=500, ttl=300)  # 5-minute cache
        self.document_cache = LRUCache(maxsize=200)  # Document cache
        self.batch_processor = BatchProcessor(batch_size=10, flush_interval=2.0)
        self._cache_stats = {'hits': 0, 'misses': 0}

    async def insert_async(self, document: Dict[str, Any]) -> int:
        """Async insert operation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.db.insert, document)

    async def search_async(self, query: Query) -> List[Document]:
        """Async search operation with caching."""
        # Generate cache key from query
        cache_key = cache_key_from_args(str(query))

        # Try cache first
        cached_result = self.query_cache.get(cache_key)
        if cached_result is not None:
            self._cache_stats['hits'] += 1
            return cached_result

        self._cache_stats['misses'] += 1
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, self.db.search, query)

        # Cache the result
        self.query_cache.set(cache_key, result)
        return result

    async def update_async(self, fields: Dict[str, Any], query: Query) -> List[int]:
        """Async update operation with cache invalidation."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, self.db.update, fields, query)

        # Invalidate relevant caches
        self.query_cache.clear()  # Simple approach - clear all
        self.document_cache.clear()

        return result

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self._cache_stats['hits'] + self._cache_stats['misses']
        return {
            'cache_hits': self._cache_stats['hits'],
            'cache_misses': self._cache_stats['misses'],
            'hit_ratio': self._cache_stats['hits'] / max(total_requests, 1),
            'query_cache_stats': self.query_cache.stats(),
            'document_cache_stats': self.document_cache.stats()
        }

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
        """Retrieve a conversation by session_id and user_id with caching."""
        cache_key = f"conv_{session_id}_{user_id}"

        # Try document cache first
        cached_conv = self.document_cache.get(cache_key)
        if cached_conv is not None:
            self._cache_stats['hits'] += 1
            return cached_conv

        self._cache_stats['misses'] += 1

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

            conversation = ConversationDocument(**data)

            # Cache the conversation
            self.document_cache.set(cache_key, conversation)

            return conversation
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
        """Append a message to an existing conversation with optimized caching."""
        cache_key = f"conv_{session_id}_{user_id}"

        conversation = await self.get_conversation(session_id, user_id)

        if conversation:
            conversation.messages.append(message)
            await self.save_conversation(conversation)

            # Update cache with modified conversation
            self.document_cache.set(cache_key, conversation)
            return True
        else:
            # Create new conversation with the message
            conversation = ConversationDocument(
                session_id=session_id,
                user_id=user_id,
                messages=[message]
            )
            await self.save_conversation(conversation)

            # Cache new conversation
            self.document_cache.set(cache_key, conversation)
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