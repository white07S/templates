"""Embeddings and vector storage using FAISS and OpenAI."""

import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import asyncio
from openai import AsyncOpenAI
import tiktoken

from config import settings
from models import Memory, SearchResult

class EmbeddingGenerator:
    """Generate embeddings using OpenAI API."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        self.dimension = settings.embedding_dimension
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return zero vector on error
            return np.zeros(self.dimension, dtype=np.float32)

    async def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts in batch."""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            embeddings = [np.array(data.embedding, dtype=np.float32) for data in response.data]
            return embeddings
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            # Return zero vectors on error
            return [np.zeros(self.dimension, dtype=np.float32) for _ in texts]

class FAISSVectorStore:
    """FAISS vector storage and search."""

    def __init__(self):
        self.dimension = settings.embedding_dimension
        self.index_path = settings.faiss_index_path / "index.faiss"
        self.metadata_path = settings.faiss_index_path / "metadata.pkl"
        self.metadata = self._load_or_create_metadata()
        self.id_counter = len(self.metadata)
        self.index = self._load_or_create_index()

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

    def save(self):
        """Save index and metadata to disk."""
        try:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            print(f"Error saving index: {e}")

    def add_vector(
        self,
        vector: np.ndarray,
        metadata: Dict[str, Any]
    ) -> int:
        """Add a single vector with metadata."""
        vector_id = self.id_counter

        # Reshape vector for FAISS
        vector = vector.reshape(1, -1)

        # Add to index
        if isinstance(self.index, faiss.IndexIVFFlat):
            # IVF index needs training
            if not self.index.is_trained:
                # Train with current vector (not ideal but works for start)
                self.index.train(vector)
            self.index.add_with_ids(vector, np.array([vector_id]))
        else:
            self.index.add(vector)

        # Store metadata
        self.metadata[vector_id] = metadata
        self.id_counter += 1

        # Save after each addition (for development/small datasets)
        # In production, you might want to batch saves
        self.save()

        return vector_id

    def add_vectors_batch(
        self,
        vectors: List[np.ndarray],
        metadata_list: List[Dict[str, Any]]
    ) -> List[int]:
        """Add multiple vectors with metadata."""
        vector_ids = []

        for vector, metadata in zip(vectors, metadata_list):
            vector_id = self.add_vector(vector, metadata)
            vector_ids.append(vector_id)

        self.save()
        return vector_ids

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter_func: Optional[callable] = None
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Search for similar vectors."""
        # Reshape query vector
        query_vector = query_vector.reshape(1, -1)

        # Search
        if self.index.ntotal == 0:
            return []

        # Search for more results if we have a filter
        search_k = k * 3 if filter_func else k
        search_k = min(search_k, self.index.ntotal)

        distances, indices = self.index.search(query_vector, search_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            metadata = self.metadata.get(int(idx), {})

            # Apply filter if provided
            if filter_func and not filter_func(metadata):
                continue

            results.append((int(idx), float(dist), metadata))

            if len(results) >= k:
                break

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
            "active_vectors": sum(1 for m in self.metadata.values() if not m.get('deleted', False))
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