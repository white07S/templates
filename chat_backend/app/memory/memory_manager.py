import asyncio
import os
import json
import pickle
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from mem0 import Memory
from uuid import UUID
import aiofiles

from ..models.schemas import MemoryItem, ChatMessage


@dataclass
class MemoryConfig:
    faiss_path: str = "./data/memory/faiss"
    memory_store_path: str = "./data/memory/memory_store.pkl"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    importance_threshold: float = 0.2
    max_profile_items: int = 20
    episodic_score_floor: float = 0.25
    memory_budget_chars: int = 350
    max_memory_char_length: int = 120


class HybridMemoryManager:
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self.memory_store: Dict[str, MemoryItem] = {}

        # Ensure directories exist
        Path(self.config.faiss_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.memory_store_path).parent.mkdir(parents=True, exist_ok=True)

        # Load existing memory store if available
        self._load_memory_store()

        # Initialize Mem0 with FAISS vector store
        self.mem0_config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.1,
                    "api_key": os.getenv("OPENAI_API_KEY")
                }
            },
            "vector_store": {
                "provider": "faiss",
                "config": {
                    "collection_name": "chat_memories",
                    "path": self.config.faiss_path,
                    "distance_strategy": "cosine"
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "api_key": os.getenv("OPENAI_API_KEY")
                }
            }
        }

        # Initialize Mem0 memory instance
        self.mem0 = Memory.from_config(self.mem0_config)

        # Load sentence transformer for local operations
        self.embedder = SentenceTransformer(self.config.embedding_model)

    def _load_memory_store(self):
        """Load memory store from disk if it exists"""
        if Path(self.config.memory_store_path).exists():
            try:
                with open(self.config.memory_store_path, 'rb') as f:
                    self.memory_store = pickle.load(f)
                print(f"Loaded {len(self.memory_store)} memories from disk")
            except Exception as e:
                print(f"Error loading memory store: {e}")
                self.memory_store = {}

    async def _save_memory_store(self):
        """Persist memory store to disk"""
        try:
            async with aiofiles.open(self.config.memory_store_path, 'wb') as f:
                await f.write(pickle.dumps(self.memory_store))
        except Exception as e:
            print(f"Error saving memory store: {e}")

    async def gate_memory(self, text: str, kind: str, importance: float) -> bool:
        """Gate memory based on quality filters"""
        if len(text) > self.config.max_memory_char_length:
            return False

        # Check for ephemeral content
        ephemeral_keywords = ["today only", "temp password", "otp", "one-time", "temporary"]
        if any(keyword in text.lower() for keyword in ephemeral_keywords):
            return False

        if importance < self.config.importance_threshold:
            return False

        return True

    async def add_memory(
        self,
        text: str,
        user_id: str,
        session_id: Optional[UUID] = None,
        kind: str = "episodic",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Add memory with gating and scoring"""

        if not await self.gate_memory(text, kind, importance):
            return None

        # Create memory using Mem0
        messages = [{"role": "user", "content": text}]

        # Add to Mem0 with user context
        result = await asyncio.to_thread(
            self.mem0.add,
            messages,
            user_id=user_id,
            metadata=metadata or {}
        )

        # Create local memory item for our hybrid system
        memory_item = MemoryItem(
            id=f"mem_{user_id}_{len(self.memory_store)}",
            text=text[:self.config.max_memory_char_length],
            kind=kind,
            user_id=user_id,
            session_id=session_id,
            created_at=datetime.now(timezone.utc),
            last_seen_at=datetime.now(timezone.utc),
            importance=importance,
            frequency=1,
            score=0.0,
            metadata=metadata
        )

        # Store in local cache
        memory_key = f"{user_id}_{memory_item.id}"
        self.memory_store[memory_key] = memory_item

        # Save to disk
        await self._save_memory_store()

        return memory_item.id

    async def retrieve_memories(
        self,
        query: str,
        user_id: str,
        k: int = 5,
        session_id: Optional[UUID] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories using hybrid approach"""

        # Use Mem0 for primary retrieval
        mem0_results = await asyncio.to_thread(
            self.mem0.search,
            query=query,
            user_id=user_id,
            limit=k
        )

        # Rescore and filter local memories
        await self._rescore_memories()

        # Combine and rank results
        combined_results = []

        for result in mem0_results:
            # Handle both dict and string responses from mem0
            if isinstance(result, dict):
                memory_text = result.get('memory', '')
                score = result.get('score', 0.5)
                metadata = result.get('metadata', {})
            else:
                # If result is a string, use it directly
                memory_text = str(result)
                score = 0.5
                metadata = {}

            combined_results.append({
                'text': memory_text,
                'score': score,
                'source': 'mem0',
                'metadata': metadata
            })

        # Add high-scoring local memories
        user_memories = [
            mem for key, mem in self.memory_store.items()
            if key.startswith(f"{user_id}_") and mem.score > self.config.episodic_score_floor
        ]

        # Sort by score and add top local memories
        user_memories.sort(key=lambda x: x.score, reverse=True)
        for mem in user_memories[:k]:
            if mem.text not in [r['text'] for r in combined_results]:
                combined_results.append({
                    'text': mem.text,
                    'score': mem.score,
                    'source': 'local',
                    'metadata': mem.metadata or {}
                })

        # Sort by score and return top k
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        return combined_results[:k]

    async def get_memory_hints(
        self,
        latest_turn: str,
        user_id: str,
        session_id: Optional[UUID] = None
    ) -> str:
        """Generate compressed memory hints for context injection"""

        memories = await self.retrieve_memories(latest_turn, user_id, k=5, session_id=session_id)

        hints = []
        used_chars = 0

        for memory in memories:
            hint_text = memory['text'][:60]  # Compress to 60 chars max
            hint = f"• {hint_text}"

            if used_chars + len(hint) + 2 > self.config.memory_budget_chars:
                break

            hints.append(hint)
            used_chars += len(hint) + 2

        if hints:
            return "[MEMORY HINTS]\n" + "\n".join(hints)
        return ""

    async def _rescore_memories(self, now: Optional[datetime] = None):
        """Rescore memories based on importance, recency, and frequency"""
        now = now or datetime.now(timezone.utc)

        for memory in self.memory_store.values():
            days_since_seen = max(0, (now - memory.last_seen_at).days)
            recency = np.exp(-days_since_seen / 30)  # 30-day half-life
            freq_score = min(1.0, np.log1p(memory.frequency) / 3)

            # Weighted scoring: 60% importance, 25% recency, 15% frequency
            memory.score = 0.6 * memory.importance + 0.25 * recency + 0.15 * freq_score

    async def garbage_collect(self):
        """Clean up low-scoring memories"""
        await self._rescore_memories()

        # Separate profile and episodic memories
        profile_memories = [
            (key, mem) for key, mem in self.memory_store.items()
            if mem.kind == "profile"
        ]
        episodic_memories = [
            (key, mem) for key, mem in self.memory_store.items()
            if mem.kind == "episodic"
        ]

        # Keep top profile memories
        profile_memories.sort(key=lambda x: x[1].score, reverse=True)
        keys_to_keep = {key for key, _ in profile_memories[:self.config.max_profile_items]}

        # Remove low-scoring episodic memories
        for key, memory in episodic_memories:
            if memory.score >= self.config.episodic_score_floor:
                keys_to_keep.add(key)

        # Remove memories not in keep list
        keys_to_remove = set(self.memory_store.keys()) - keys_to_keep
        for key in keys_to_remove:
            del self.memory_store[key]

        # Save after garbage collection
        await self._save_memory_store()

    async def update_memory_frequency(self, memory_id: str, user_id: str):
        """Update memory frequency when accessed"""
        memory_key = f"{user_id}_{memory_id}"
        if memory_key in self.memory_store:
            memory = self.memory_store[memory_key]
            memory.frequency += 1
            memory.last_seen_at = datetime.now(timezone.utc)

    async def get_user_memory_stats(self, user_id: str) -> Dict[str, int]:
        """Get memory statistics for a user"""
        user_memories = [
            mem for key, mem in self.memory_store.items()
            if key.startswith(f"{user_id}_")
        ]

        profile_count = len([m for m in user_memories if m.kind == "profile"])
        episodic_count = len([m for m in user_memories if m.kind == "episodic"])

        return {
            "total": len(user_memories),
            "profile": profile_count,
            "episodic": episodic_count
        }

    async def extract_and_store_from_conversation(
        self,
        user_message: str,
        assistant_message: str,
        user_id: str,
        session_id: Optional[UUID] = None
    ):
        """Extract and store important information from conversation"""
        # Store the user query as episodic memory
        user_memory = f"User asked: {user_message[:100]}"
        await self.add_memory(
            text=user_memory,
            user_id=user_id,
            session_id=session_id,
            kind="episodic",
            importance=0.4,
            metadata={"type": "user_query"}
        )

        # Keywords that indicate important information to remember
        importance_indicators = [
            "remember", "my name", "i prefer", "i like", "i hate", "allergic",
            "birthday", "job", "work", "live in", "from now on", "always",
            "never", "important", "note", "deadline", "meeting", "appointment"
        ]

        # Check for important information in user message
        user_lower = user_message.lower()
        importance = 0.3

        for indicator in importance_indicators:
            if indicator in user_lower:
                importance = 0.8
                break

        # If important, store the full context
        if importance > 0.5:
            memory_text = user_message[:self.config.max_memory_char_length]
            await self.add_memory(
                text=memory_text,
                user_id=user_id,
                session_id=session_id,
                kind="profile" if "prefer" in user_lower or "always" in user_lower else "episodic",
                importance=importance,
                metadata={"source": "conversation"}
            )

        # Also check assistant response for commitments or important info
        if any(word in assistant_message.lower() for word in ["will remember", "noted", "i'll", "i will"]):
            commitment = f"Assistant committed: {assistant_message[:100]}"
            await self.add_memory(
                text=commitment,
                user_id=user_id,
                session_id=session_id,
                kind="episodic",
                importance=0.6,
                metadata={"type": "assistant_commitment"}
            )