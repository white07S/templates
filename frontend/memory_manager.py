"""Memory management system for dual-level memory architecture."""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
from datetime import datetime
from collections import deque
import json
from openai import AsyncOpenAI
import tiktoken

from config import settings
from models import (
    Memory, MemoryType, Message, MessageRole,
    ConversationDocument, TokenBudget, SearchResult
)
from storage import ConversationStore, MemoryStore
from search import HybridRetriever
from embeddings import EmbeddingGenerator, VectorMemoryStore

class MemoryCompressor:
    """Handles memory compression and extraction."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

    async def compress_text(self, text: str, target_ratio: float = 5.0) -> str:
        """Compress text using LLM while preserving semantic meaning."""
        if len(text) < 100:  # Don't compress short text
            return text

        prompt = f"""Compress the following text to approximately {int(100/target_ratio)}% of its original length while preserving all key information and semantic meaning.
        Focus on removing redundancy and verbose language while keeping facts, entities, and relationships intact.

        Original text:
        {text}

        Compressed version:"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a text compression expert. Preserve semantic meaning while reducing verbosity."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=int(len(self.tokenizer.encode(text)) / target_ratio),
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error compressing text: {e}")
            return text[:int(len(text) / target_ratio)]  # Fallback to truncation

    async def extract_memories(self, conversation: List[Message]) -> List[Memory]:
        """Extract memories from a conversation."""
        if not conversation:
            return []

        # Format conversation for extraction
        conv_text = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in conversation
        ])

        # Check if we have a valid API key
        if settings.openai_api_key == "sk-test-key-placeholder":
            print("⚠️  Using fallback memory extraction (no valid OpenAI API key)")
            return self._extract_memories_fallback(conversation)

        prompt = f"""Extract important memories from this conversation. Identify:
        1. Semantic memories (facts, information, knowledge)
        2. Episodic memories (events, experiences, stories)
        3. Procedural memories (preferences, instructions, how-to)

        Format as JSON array with: type, content, importance (0-1), keywords

        Conversation:
        {conv_text}

        Memories JSON:"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a memory extraction expert. Extract distinct, valuable memories."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.5,
                response_format={ "type": "json_object" }
            )

            # Parse JSON response
            memories_data = json.loads(response.choices[0].message.content)
            memories = []

            for mem_data in memories_data.get("memories", []):
                memory_type = MemoryType(mem_data.get("type", "semantic"))
                content = mem_data.get("content", "")
                importance = mem_data.get("importance", 0.5)
                keywords = mem_data.get("keywords", [])

                if content:
                    memories.append(Memory(
                        user_id="",  # Will be set by caller
                        memory_type=memory_type,
                        content=content,
                        importance_score=importance,
                        keywords=keywords
                    ))

            return memories
        except Exception as e:
            print(f"Error extracting memories with OpenAI, using fallback: {e}")
            return self._extract_memories_fallback(conversation)

    def _extract_memories_fallback(self, conversation: List[Message]) -> List[Memory]:
        """Fallback memory extraction without LLM."""
        memories = []

        # Simple heuristic-based extraction
        for msg in conversation[-3:]:  # Only look at recent messages
            if msg.role == MessageRole.USER and len(msg.content) > 20:
                # Extract potential facts or preferences
                content = msg.content.strip()

                # Look for personal information patterns
                if any(keyword in content.lower() for keyword in ["my name is", "i am", "i like", "i love", "i prefer"]):
                    memories.append(Memory(
                        user_id="",
                        memory_type=MemoryType.SEMANTIC,
                        content=content,
                        importance_score=0.8,
                        keywords=content.lower().split()[:5]
                    ))
                elif any(keyword in content.lower() for keyword in ["tell me", "what", "how", "explain"]):
                    # User asked a question - less important but still worth remembering
                    memories.append(Memory(
                        user_id="",
                        memory_type=MemoryType.EPISODIC,
                        content=f"User asked: {content}",
                        importance_score=0.5,
                        keywords=content.lower().split()[:3]
                    ))

        print(f"📝 Extracted {len(memories)} memories using fallback method")
        return memories

class SessionMemory:
    """Manages session-level conversation memory."""

    def __init__(self, session_id: str, user_id: str):
        self.session_id = session_id
        self.user_id = user_id
        self.conversation_buffer = deque(maxlen=100)  # Keep last 100 messages
        self.active_entities = {}
        self.context_summary = ""
        # Use user-specific conversation store
        self.conversation_store = ConversationStore(user_id=user_id)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.embedding_generator = EmbeddingGenerator()
        self._loaded = False
        self._embedding_counter = 0

    async def _load_existing_conversation(self):
        """Load existing conversation from storage."""
        if self._loaded:
            return

        # Load existing conversation
        conversation = await self.conversation_store.get_conversation(
            self.session_id, self.user_id
        )

        if conversation and conversation.messages:
            # Load recent messages into buffer
            for message in conversation.messages[-100:]:  # Last 100 messages
                self.conversation_buffer.append(message)

            # Load memory state
            if conversation.memory_state:
                self.context_summary = conversation.memory_state.get('context_summary', '')
                self.active_entities = conversation.memory_state.get('active_entities', {})

        self._loaded = True

    async def add_message(self, message: Message):
        """Add a message to the session."""
        # Generate embedding for the message (only if not placeholder API key)
        if (settings.openai_api_key != "sk-test-key-placeholder"
            and len(message.content.strip()) > 10):
            try:
                embedding = await self.embedding_generator.generate_embedding(message.content)
                message.embedding_id = self._embedding_counter
                self._embedding_counter += 1
                print(f"🔗 Generated embedding {message.embedding_id} for message: {message.content[:50]}...")
            except Exception as e:
                print(f"⚠️  Failed to generate embedding: {e}")
                message.embedding_id = None
        else:
            message.embedding_id = None

        self.conversation_buffer.append(message)

        # Save to persistent storage
        await self.conversation_store.append_message(
            self.session_id,
            self.user_id,
            message
        )

    async def get_recent_context(
        self,
        token_budget: int,
        include_summary: bool = True
    ) -> Tuple[List[Message], int]:
        """Get recent conversation within token budget."""
        # Load existing conversation if not already loaded
        await self._load_existing_conversation()

        messages = []
        total_tokens = 0

        # Add context summary if requested
        if include_summary and self.context_summary:
            summary_tokens = len(self.tokenizer.encode(self.context_summary))
            if summary_tokens < token_budget / 2:
                messages.append(Message(
                    role=MessageRole.SYSTEM,
                    content=f"Previous context: {self.context_summary}"
                ))
                total_tokens += summary_tokens

        # Add recent messages in reverse order
        for message in reversed(list(self.conversation_buffer)):
            msg_tokens = len(self.tokenizer.encode(message.content))

            if total_tokens + msg_tokens > token_budget:
                break

            messages.insert(0, message)  # Insert at beginning to maintain order
            total_tokens += msg_tokens

        return messages, total_tokens

    async def update_context_summary(self, messages: List[Message]):
        """Update the rolling context summary."""
        if len(messages) < 5:  # Don't summarize very short conversations
            return

        # Only summarize older messages
        messages_to_summarize = messages[:-3]  # Keep last 3 messages verbatim

        if not messages_to_summarize:
            return

        conv_text = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in messages_to_summarize
        ])

        client = AsyncOpenAI(api_key=settings.openai_api_key)
        prompt = f"""Summarize this conversation focusing on key topics, decisions, and context needed for continuation:

        {conv_text}

        Concise summary:"""

        try:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Create concise conversation summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            self.context_summary = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error updating context summary: {e}")

class UserMemory:
    """Manages user-level persistent memory."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        # Use user-specific memory store
        self.memory_store = MemoryStore(user_id=user_id)
        self.retriever = HybridRetriever()
        self.compressor = MemoryCompressor()

    async def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        importance_score: float = 0.5,
        source_session_id: Optional[str] = None,
        keywords: Optional[List[str]] = None
    ) -> Memory:
        """Add a new memory."""
        # Compress content if it's long
        compressed_content = None
        if len(content) > 500:
            compressed_content = await self.compressor.compress_text(
                content,
                settings.memory_compression_ratio
            )

        # Create memory object
        memory = Memory(
            user_id=self.user_id,
            memory_type=memory_type,
            content=content,
            compressed_content=compressed_content,
            importance_score=importance_score,
            source_session_id=source_session_id,
            keywords=keywords or []
        )

        # Save to storage
        await self.memory_store.save_memory(memory)

        # Add to search indices
        await self.retriever.add_memory_to_indices(memory)

        return memory

    async def search_memories(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[MemoryType] = None
    ) -> List[Memory]:
        """Search user memories."""
        # Use hybrid retriever for search
        search_results = await self.retriever.hybrid_search(
            query=query,
            user_id=self.user_id,
            limit=limit
        )

        # Load full memory objects
        memories = []
        for result in search_results:
            memory_id = result.metadata.get('memory_id')
            if memory_id:
                memory = await self.memory_store.get_memory(memory_id)
                if memory:
                    # Update access metadata
                    await self.memory_store.update_memory_access(memory_id)
                    memories.append(memory)

        return memories

    async def get_relevant_memories(
        self,
        context: str,
        token_budget: int
    ) -> Tuple[List[Memory], int]:
        """Get relevant memories within token budget."""
        # Search for relevant memories
        memories = await self.search_memories(context, limit=20)

        selected_memories = []
        total_tokens = 0
        tokenizer = tiktoken.encoding_for_model("gpt-4")

        for memory in memories:
            # Use compressed content if available
            content = memory.compressed_content or memory.content
            memory_tokens = len(tokenizer.encode(content))

            if total_tokens + memory_tokens > token_budget:
                break

            selected_memories.append(memory)
            total_tokens += memory_tokens

        return selected_memories, total_tokens

    async def consolidate_memories(self):
        """Consolidate and deduplicate memories."""
        # Get all memories
        all_memories = await self.memory_store.get_user_memories(
            self.user_id,
            limit=1000
        )

        if len(all_memories) < 10:
            return  # Not enough memories to consolidate

        # Group by memory type
        by_type = {}
        for memory in all_memories:
            if memory.memory_type not in by_type:
                by_type[memory.memory_type] = []
            by_type[memory.memory_type].append(memory)

        # Consolidate each type
        for memory_type, memories in by_type.items():
            if len(memories) < 5:
                continue

            # Find similar memories to merge
            # This would use embedding similarity in production
            # For now, we'll skip actual consolidation
            pass

class MemoryOrchestrator:
    """Orchestrates session and user memory operations."""

    def __init__(self):
        self.session_memories = {}  # session_id -> SessionMemory
        self.user_memories = {}  # user_id -> UserMemory
        self.compressor = MemoryCompressor()

    def get_session_memory(self, session_id: str, user_id: str) -> SessionMemory:
        """Get or create session memory."""
        if session_id not in self.session_memories:
            self.session_memories[session_id] = SessionMemory(session_id, user_id)
        return self.session_memories[session_id]

    def get_user_memory(self, user_id: str) -> UserMemory:
        """Get or create user memory."""
        if user_id not in self.user_memories:
            self.user_memories[user_id] = UserMemory(user_id)
        return self.user_memories[user_id]

    def calculate_token_budget(
        self,
        total_budget: int = None
    ) -> TokenBudget:
        """Calculate dynamic token budget allocation."""
        if total_budget is None:
            total_budget = settings.model_context_window

        # Reserve tokens for system prompt
        system_tokens = settings.system_prompt_tokens

        # Calculate allocations
        remaining = total_budget - system_tokens

        if total_budget <= 8192:  # Small models
            conversation_tokens = int(remaining * 0.7)
            memory_tokens = int(remaining * 0.3)
        elif total_budget <= 32768:  # Medium models
            conversation_tokens = int(remaining * 0.6)
            memory_tokens = int(remaining * 0.4)
        else:  # Large models
            conversation_tokens = int(remaining * 0.5)
            memory_tokens = int(remaining * 0.5)

        return TokenBudget(
            conversation_tokens=conversation_tokens,
            memory_tokens=memory_tokens,
            system_tokens=system_tokens,
            total_budget=total_budget
        )

    async def process_conversation_for_memories(
        self,
        session_id: str,
        user_id: str
    ):
        """Extract and store memories from a conversation."""
        print(f"🧠 Processing memories for session {session_id}, user {user_id}")

        # Get conversation using user-specific store
        store = ConversationStore(user_id=user_id)
        conversation = await store.get_conversation(session_id, user_id)

        if not conversation:
            print(f"❌ No conversation found for session {session_id}")
            return

        if len(conversation.messages) < 3:
            print(f"⚠️  Not enough messages ({len(conversation.messages)}) to extract memories")
            return

        print(f"📥 Found {len(conversation.messages)} messages, extracting memories...")

        # Extract memories
        memories = await self.compressor.extract_memories(conversation.messages)
        print(f"📝 Extracted {len(memories)} memories")

        if not memories:
            print("⚠️  No memories extracted from conversation")
            return

        # Store memories
        user_memory = self.get_user_memory(user_id)
        stored_count = 0

        for memory in memories:
            memory.user_id = user_id
            memory.source_session_id = session_id

            try:
                stored_memory = await user_memory.add_memory(
                    content=memory.content,
                    memory_type=memory.memory_type,
                    importance_score=memory.importance_score,
                    source_session_id=session_id,
                    keywords=memory.keywords
                )
                stored_count += 1
                print(f"💾 Stored memory {stored_memory.memory_id}: {memory.content[:50]}...")
            except Exception as e:
                print(f"❌ Failed to store memory: {e}")

        print(f"✅ Successfully stored {stored_count}/{len(memories)} memories")