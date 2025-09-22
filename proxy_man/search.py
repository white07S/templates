"""Hybrid search implementation using Tantivy and FAISS."""

import tantivy
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import asyncio
from datetime import datetime
import math
import numpy as np
from rake_nltk import Rake
from collections import defaultdict

from config import settings
from models import SearchResult, Memory, IntentType
from embeddings import VectorMemoryStore

class TantivySearchEngine:
    """Full-text search using Tantivy."""

    def __init__(self):
        self.index_path = str(settings.tantivy_index_path)
        self.schema = self._build_schema()
        self.index = self._load_or_create_index()
        self.rake = Rake()

    def _build_schema(self) -> tantivy.Schema:
        """Build the Tantivy schema."""
        schema_builder = tantivy.SchemaBuilder()

        # Add fields
        schema_builder.add_text_field("content", stored=True, tokenizer_name='en_stem')
        schema_builder.add_text_field("compressed_content", stored=True, tokenizer_name='en_stem')
        schema_builder.add_text_field("keywords", stored=True)
        schema_builder.add_text_field("user_id", stored=True)
        schema_builder.add_text_field("session_id", stored=True)
        schema_builder.add_text_field("memory_id", stored=True)
        schema_builder.add_text_field("memory_type", stored=True)
        schema_builder.add_date_field("timestamp", stored=True)
        schema_builder.add_float_field("importance_score", stored=True, indexed=True)

        return schema_builder.build()

    def _load_or_create_index(self) -> tantivy.Index:
        """Load existing index or create new one."""
        try:
            # Try to open existing index
            index = tantivy.Index(self.schema, path=self.index_path)
            return index
        except Exception as e:
            print(f"Creating new Tantivy index: {e}")
            # Create new index
            Path(self.index_path).mkdir(parents=True, exist_ok=True)
            index = tantivy.Index(self.schema, path=self.index_path)
            return index

    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords using RAKE."""
        try:
            self.rake.extract_keywords_from_text(text)
            keywords = self.rake.get_ranked_phrases()[:10]  # Top 10 keywords
            return keywords
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            # Fallback to simple word extraction
            words = text.lower().split()
            return list(set(words))[:10]

    def add_document(
        self,
        content: str,
        user_id: str,
        memory_id: str,
        memory_type: str,
        session_id: Optional[str] = None,
        compressed_content: Optional[str] = None,
        importance_score: float = 0.5,
        timestamp: Optional[datetime] = None
    ):
        """Add a document to the index."""
        writer = self.index.writer()

        # Extract keywords
        keywords = self.extract_keywords(content)

        # Build document
        doc = tantivy.Document()
        doc.add_text("content", content)
        doc.add_text("user_id", user_id)
        doc.add_text("memory_id", memory_id)
        doc.add_text("memory_type", memory_type)
        doc.add_text("keywords", " ".join(keywords))

        if compressed_content:
            doc.add_text("compressed_content", compressed_content)

        if session_id:
            doc.add_text("session_id", session_id)

        doc.add_float("importance_score", importance_score)

        if timestamp:
            doc.add_date("timestamp", timestamp)

        writer.add_document(doc)
        writer.commit()

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 20
    ) -> List[SearchResult]:
        """Search the index."""
        searcher = self.index.searcher()

        # Build query
        query_parts = []

        # Add content query
        query_parts.append(f'content:"{query}"')

        # Add filters
        if user_id:
            query_parts.append(f'user_id:{user_id}')

        if session_id:
            query_parts.append(f'session_id:{session_id}')

        # Join query parts
        full_query = " AND ".join(query_parts) if len(query_parts) > 1 else query_parts[0]

        # Parse and execute query
        try:
            query_obj = self.index.parse_query(full_query, ["content", "compressed_content", "keywords"])
            search_results = searcher.search(query_obj, limit).hits
        except Exception as e:
            print(f"Error parsing query: {e}")
            # Fallback to simple content search
            query_obj = self.index.parse_query(query, ["content"])
            search_results = searcher.search(query_obj, limit).hits

        # Convert to SearchResult objects
        results = []
        for score, doc_address in search_results:
            doc = searcher.doc(doc_address)

            # Extract fields
            content = doc.get_first("content") or ""
            memory_id = doc.get_first("memory_id") or ""
            memory_type = doc.get_first("memory_type") or ""
            importance = doc.get_first("importance_score") or 0.5

            metadata = {
                'memory_id': memory_id,
                'memory_type': memory_type,
                'importance_score': importance,
                'tantivy_score': score
            }

            results.append(SearchResult(
                content=content,
                score=score,
                source="lexical",
                metadata=metadata
            ))

        return results

    def update_document(
        self,
        memory_id: str,
        content: str,
        compressed_content: Optional[str] = None
    ):
        """Update an existing document."""
        # Tantivy doesn't support updates directly, so we delete and re-add
        writer = self.index.writer()

        # Delete old document
        writer.delete_term(self.schema.get_field("memory_id").unwrap(), memory_id)

        # Commit deletion
        writer.commit()

        # Add updated document (would need full metadata in practice)
        # This is simplified - in production, you'd retrieve the full document first
        self.add_document(
            content=content,
            user_id="",  # Would need to retrieve this
            memory_id=memory_id,
            memory_type="",  # Would need to retrieve this
            compressed_content=compressed_content
        )

class HybridRetriever:
    """Combines lexical and semantic search with Reciprocal Rank Fusion."""

    def __init__(self):
        self.tantivy_engine = TantivySearchEngine()
        self.vector_store = VectorMemoryStore()
        self.intent_weights = {
            IntentType.META_QUERY: {"relevance": 0.3, "recency": 0.7},
            IntentType.SEMANTIC_SEARCH: {"relevance": 0.8, "recency": 0.2},
            IntentType.TEMPORAL_QUERY: {"relevance": 0.4, "recency": 0.6},
            IntentType.MEMORY_RETRIEVAL: {"relevance": 0.6, "recency": 0.4},
            IntentType.GENERAL: {"relevance": 0.5, "recency": 0.5}
        }

    def classify_intent(self, query: str) -> IntentType:
        """Classify the intent of a query."""
        query_lower = query.lower()

        # Meta query patterns
        meta_patterns = ["previous", "last", "earlier", "before", "ago", "what did i", "what was"]
        if any(pattern in query_lower for pattern in meta_patterns):
            return IntentType.META_QUERY

        # Temporal query patterns
        temporal_patterns = ["when", "yesterday", "today", "week", "month", "date", "time"]
        if any(pattern in query_lower for pattern in temporal_patterns):
            return IntentType.TEMPORAL_QUERY

        # Memory retrieval patterns
        memory_patterns = ["remember", "recall", "memory", "forgot", "remind"]
        if any(pattern in query_lower for pattern in memory_patterns):
            return IntentType.MEMORY_RETRIEVAL

        # Semantic search patterns
        semantic_patterns = ["about", "related", "similar", "like", "regarding"]
        if any(pattern in query_lower for pattern in semantic_patterns):
            return IntentType.SEMANTIC_SEARCH

        return IntentType.GENERAL

    def compute_temporal_score(self, timestamp: datetime, decay_lambda: float = 0.5) -> float:
        """Compute temporal relevance score with exponential decay."""
        if not timestamp:
            return 0.0

        time_diff_days = (datetime.utcnow() - timestamp).total_seconds() / 86400
        return math.exp(-decay_lambda * time_diff_days)

    def reciprocal_rank_fusion(
        self,
        result_lists: List[List[SearchResult]],
        k: int = 60
    ) -> List[SearchResult]:
        """Perform Reciprocal Rank Fusion on multiple result lists."""
        fused_scores = defaultdict(float)
        result_map = {}

        for result_list in result_lists:
            for rank, result in enumerate(result_list, 1):
                # Use content as key for deduplication
                key = result.metadata.get('memory_id', result.content[:100])
                fused_scores[key] += 1.0 / (k + rank)

                # Keep the result object
                if key not in result_map:
                    result_map[key] = result

        # Sort by fused score
        sorted_results = sorted(
            [(key, score) for key, score in fused_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )

        # Return SearchResults with updated scores
        final_results = []
        for key, fused_score in sorted_results:
            result = result_map[key]
            result.score = fused_score
            result.source = "hybrid"
            final_results.append(result)

        return final_results

    async def hybrid_search(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        limit: int = 20
    ) -> List[SearchResult]:
        """Perform hybrid search combining lexical and semantic search."""
        # Classify intent
        intent = self.classify_intent(query)

        # Parallel search
        lexical_task = asyncio.create_task(
            asyncio.to_thread(
                self.tantivy_engine.search,
                query, user_id, session_id, limit
            )
        )

        semantic_task = self.vector_store.search_memories(
            query, user_id, limit
        )

        # Wait for both searches
        lexical_results, semantic_results = await asyncio.gather(
            lexical_task, semantic_task
        )

        # Apply Reciprocal Rank Fusion
        fused_results = self.reciprocal_rank_fusion(
            [lexical_results, semantic_results],
            k=settings.hybrid_search_k
        )

        # Apply temporal ranking based on intent
        weights = self.intent_weights[intent]
        for result in fused_results:
            # Get timestamp from metadata
            timestamp_str = result.metadata.get('created_at')
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
                temporal_score = self.compute_temporal_score(
                    timestamp,
                    settings.temporal_decay_lambda
                )
            else:
                temporal_score = 0.5

            # Combine relevance and temporal scores
            importance = result.metadata.get('importance_score', 0.5)
            final_score = (
                weights["relevance"] * result.score +
                weights["recency"] * temporal_score +
                importance * 0.2  # Importance boost
            )
            result.score = final_score

        # Re-sort by final score
        fused_results.sort(key=lambda x: x.score, reverse=True)

        return fused_results[:limit]

    async def add_memory_to_indices(self, memory: Memory):
        """Add a memory to both search indices."""
        # Add to vector store
        await self.vector_store.add_memory(memory)

        # Add to Tantivy
        self.tantivy_engine.add_document(
            content=memory.content,
            user_id=memory.user_id,
            memory_id=memory.memory_id,
            memory_type=memory.memory_type,
            session_id=memory.source_session_id,
            compressed_content=memory.compressed_content,
            importance_score=memory.importance_score,
            timestamp=memory.created_at
        )