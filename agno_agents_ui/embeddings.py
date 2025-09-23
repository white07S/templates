"""Embeddings and vector storage using FAISS and OpenAI with performance optimizations."""

import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import asyncio
from openai import AsyncOpenAI
import tiktoken
import time
import signal
import atexit
from threading import Timer

from config import settings
from models import Memory, SearchResult
from optimized_openai import OptimizedEmbeddingClient
from cache_utils import BatchProcessor, LRUCache

class EmbeddingGenerator:
    """Generate embeddings using optimized OpenAI API client."""

    def __init__(self):
        self.optimized_client = OptimizedEmbeddingClient()
        self.dimension = settings.embedding_dimension

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.optimized_client.tokenizer.encode(text))

    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text with caching."""
        return await self.optimized_client.generate_embedding(text)

    async def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts with intelligent batching."""
        return await self.optimized_client.generate_embeddings_batch(texts)

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.optimized_client.get_stats()

class FAISSVectorStore:
    """FAISS vector storage and search with optimized batch operations."""

    def __init__(self):
        self.dimension = settings.embedding_dimension
        self.index_path = settings.faiss_index_path / "index.faiss"
        self.metadata_path = settings.faiss_index_path / "metadata.pkl"
        self.metadata = self._load_or_create_metadata()
        self.id_counter = len(self.metadata)
        self.index = self._load_or_create_index()

        # Batch processing for performance
        self.batch_processor = BatchProcessor(batch_size=50, flush_interval=10.0)
        self._pending_vectors = []
        self._pending_metadata = []

        # Performance optimizations
        self.search_cache = LRUCache(maxsize=100)  # Cache search results
        self._last_save = time.time()
        self._save_interval = 30.0  # Save every 30 seconds
        self._dirty = False

        # Setup auto-save and cleanup
        self._setup_auto_save()
        self._setup_cleanup_handlers()

    def _load_or_create_index(self) -> faiss.Index:
        """Load existing index or create new one."""
        if self.index_path.exists():
            try:
                index = faiss.read_index(str(self.index_path))
                return index
            except Exception as e:
                print(f"Error loading index, creating new: {e}")

        # Create new index
        if self.id_counter > 10000:
            # Use IVF for large datasets
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(
                quantizer,
                self.dimension,
                settings.faiss_nclusters
            )
            # Need to train IVF index with some data
            index.nprobe = settings.faiss_nprobe
        else:
            # Use flat index for small datasets
            index = faiss.IndexFlatL2(self.dimension)

        return index

    def _load_or_create_metadata(self) -> Dict[int, Dict[str, Any]]:
        """Load existing metadata or create new one."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading metadata, creating new: {e}")
        return {}

    def _setup_auto_save(self):
        """Setup automatic saving timer."""
        def auto_save():
            if self._dirty and time.time() - self._last_save > self._save_interval:
                self.save()
            # Schedule next auto-save
            Timer(self._save_interval, auto_save).start()

        Timer(self._save_interval, auto_save).start()

    def _setup_cleanup_handlers(self):
        """Setup cleanup handlers for safe shutdown."""
        def cleanup_handler(signum=None, frame=None):
            self.force_save()

        # Register cleanup for various exit scenarios
        atexit.register(cleanup_handler)
        signal.signal(signal.SIGINT, cleanup_handler)
        signal.signal(signal.SIGTERM, cleanup_handler)

    def save(self):
        """Save index and metadata to disk."""
        try:
            # Process any pending batches first
            if self._pending_vectors:
                self._process_pending_batch()

            faiss.write_index(self.index, str(self.index_path))
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)

            self._last_save = time.time()
            self._dirty = False
            print(f"✅ FAISS index saved with {self.index.ntotal} vectors")

        except Exception as e:
            print(f"❌ Error saving index: {e}")

    def force_save(self):
        """Force immediate save of all pending data."""
        print("🔄 Force saving FAISS index...")
        self.save()

    def add_vector(
        self,
        vector: np.ndarray,
        metadata: Dict[str, Any]
    ) -> int:
        """Add a single vector with metadata using batch processing."""
        vector_id = self.id_counter
        self.id_counter += 1

        # Add to pending batch
        self._pending_vectors.append((vector_id, vector))
        self._pending_metadata.append((vector_id, metadata))
        self._dirty = True

        # Store metadata immediately for consistency
        self.metadata[vector_id] = metadata

        # Process batch if it's full or force save
        if len(self._pending_vectors) >= 50:  # batch_size
            self._process_pending_batch()

        return vector_id

    def _process_pending_batch(self):
        """Process pending vectors in batch."""
        if not self._pending_vectors:
            return

        print(f"🔄 Processing batch of {len(self._pending_vectors)} vectors...")

        try:
            # Prepare batch data
            vectors = []
            vector_ids = []

            for vector_id, vector in self._pending_vectors:
                vectors.append(vector.reshape(1, -1))
                vector_ids.append(vector_id)

            # Combine all vectors
            if vectors:
                batch_vectors = np.vstack(vectors)

                # Add to index
                if isinstance(self.index, faiss.IndexIVFFlat):
                    # IVF index needs training
                    if not self.index.is_trained:
                        # Train with batch data
                        self.index.train(batch_vectors)
                    self.index.add_with_ids(batch_vectors, np.array(vector_ids))
                else:
                    self.index.add(batch_vectors)

                print(f"✅ Added {len(vectors)} vectors to FAISS index")

            # Clear pending batches
            self._pending_vectors.clear()
            self._pending_metadata.clear()

            # Clear search cache since index changed
            self.search_cache.clear()

        except Exception as e:
            print(f"❌ Error processing vector batch: {e}")
            # Don't clear pending data on error - will retry later

    def add_vectors_batch(
        self,
        vectors: List[np.ndarray],
        metadata_list: List[Dict[str, Any]]
    ) -> List[int]:
        """Add multiple vectors with metadata efficiently."""
        vector_ids = []

        # Add all vectors to pending batch
        for vector, metadata in zip(vectors, metadata_list):
            vector_id = self.id_counter
            self.id_counter += 1

            self._pending_vectors.append((vector_id, vector))
            self._pending_metadata.append((vector_id, metadata))
            self.metadata[vector_id] = metadata
            vector_ids.append(vector_id)

        self._dirty = True

        # Process the entire batch immediately
        self._process_pending_batch()

        return vector_ids

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter_func: Optional[callable] = None
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Search for similar vectors with caching."""
        # Process pending vectors first to ensure complete results
        if self._pending_vectors:
            self._process_pending_batch()

        # Generate cache key
        import hashlib
        vector_hash = hashlib.md5(query_vector.tobytes()).hexdigest()
        filter_hash = str(hash(str(filter_func))) if filter_func else "no_filter"
        cache_key = f"{vector_hash}_{k}_{filter_hash}"

        # Try cache first
        cached_result = self.search_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Reshape query vector
        query_vector = query_vector.reshape(1, -1)

        # Early exit for empty index
        if self.index.ntotal == 0:
            return []

        # Search for more results if we have a filter
        search_k = k * 3 if filter_func else k
        search_k = min(search_k, self.index.ntotal)

        try:
            distances, indices = self.index.search(query_vector, search_k)
        except Exception as e:
            print(f"❌ FAISS search error: {e}")
            return []

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            metadata = self.metadata.get(int(idx), {})

            # Skip deleted vectors
            if metadata.get('deleted', False):
                continue

            # Apply filter if provided
            if filter_func and not filter_func(metadata):
                continue

            results.append((int(idx), float(dist), metadata))

            if len(results) >= k:
                break

        # Cache results
        self.search_cache.set(cache_key, results)

        return results

    def update_metadata(self, vector_id: int, metadata: Dict[str, Any]):
        """Update metadata for a vector."""
        if vector_id in self.metadata:
            self.metadata[vector_id] = metadata
            self.save()

    def remove_vector(self, vector_id: int):
        """Remove a vector (mark as deleted in metadata)."""
        if vector_id in self.metadata:
            self.metadata[vector_id]['deleted'] = True
            self.save()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": type(self.index).__name__,
            "metadata_count": len(self.metadata),
            "active_vectors": sum(1 for m in self.metadata.values() if not m.get('deleted', False)),
            "pending_vectors": len(self._pending_vectors),
            "search_cache_stats": self.search_cache.stats(),
            "last_save": self._last_save,
            "dirty": self._dirty
        }

class VectorMemoryStore:
    """High-level interface for vector-based memory storage."""

    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = FAISSVectorStore()

    async def add_memory(self, memory: Memory) -> int:
        """Add a memory to the vector store."""
        # Check if we have a valid API key
        if settings.openai_api_key == "sk-test-key-placeholder":
            print(f"⚠️  Skipping vector storage for memory {memory.memory_id} (no valid API key)")
            return -1

        try:
            # Generate embedding
            embedding = await self.embedding_generator.generate_embedding(
                memory.compressed_content or memory.content
            )

            # Prepare metadata
            metadata = {
                'memory_id': memory.memory_id,
                'user_id': memory.user_id,
                'memory_type': memory.memory_type,
                'importance_score': memory.importance_score,
                'created_at': memory.created_at.isoformat(),
                'keywords': memory.keywords,
                'source_session_id': memory.source_session_id
            }

            # Add to vector store
            vector_id = self.vector_store.add_vector(embedding, metadata)
            print(f"✅ Added memory {memory.memory_id} to FAISS index with vector_id {vector_id}")

            # Update memory with FAISS ID in the database
            memory.faiss_id = vector_id

        except Exception as e:
            print(f"❌ Failed to add memory {memory.memory_id} to vector store: {e}")
            import traceback
            traceback.print_exc()
            return -1

        return vector_id

    async def search_memories(
        self,
        query: str,
        user_id: str,
        k: int = 10,
        memory_type: Optional[str] = None
    ) -> List[SearchResult]:
        """Search for similar memories."""
        # Generate query embedding
        query_embedding = await self.embedding_generator.generate_embedding(query)

        # Create filter function
        def filter_func(metadata: Dict[str, Any]) -> bool:
            if metadata.get('deleted', False):
                return False
            if metadata.get('user_id') != user_id:
                return False
            if memory_type and metadata.get('memory_type') != memory_type:
                return False
            return True

        # Search
        results = self.vector_store.search(
            query_embedding,
            k=k,
            filter_func=filter_func
        )

        # Convert to SearchResult
        search_results = []
        for vector_id, distance, metadata in results:
            # Convert L2 distance to similarity score (0-1)
            similarity = 1.0 / (1.0 + distance)

            search_results.append(SearchResult(
                content=f"Memory {metadata['memory_id']}",
                score=similarity,
                source="semantic",
                metadata=metadata
            ))

        return search_results

    async def update_memory_embedding(self, memory: Memory):
        """Update embedding for an existing memory."""
        if memory.faiss_id is not None:
            # Mark old vector as deleted
            self.vector_store.remove_vector(memory.faiss_id)

        # Add new vector
        await self.add_memory(memory)