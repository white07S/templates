"""
Core search engine module implementing keyword, semantic, and hybrid search.
"""
import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging

# Tantivy imports
try:
    import tantivy
except ImportError:
    print("Please install tantivy-py: uv add tantivy")
    raise

# FAISS imports
try:
    import faiss
except ImportError:
    print("Please install faiss-cpu: uv add faiss-cpu")
    raise

from config import SearchConfig
from utils import (
    OptimizedEmbeddingClient,
    RRFScorer,
    clean_text,
    logger
)


class TantivySearcher:
    """Tantivy-based keyword search engine."""
    
    def __init__(self, index_path: str = SearchConfig.TANTIVY_INDEX_PATH):
        self.index_path = Path(index_path)
        self.index = None
        self.searcher = None
        self.schema = None
        self._load_index()
    
    def _load_index(self):
        """Load existing Tantivy index."""
        if not self.index_path.exists():
            raise ValueError(f"Tantivy index not found at {self.index_path}")
        
        # Load schema
        meta_file = self.index_path / "meta.json"
        if not meta_file.exists():
            raise ValueError(f"Index metadata not found at {meta_file}")
        
        # Open index
        self.index = tantivy.Index.open(str(self.index_path))
        self.index.reload()
        self.searcher = self.index.searcher()
        
        logger.info(f"Loaded Tantivy index from {self.index_path}")
    
    def search(self, 
               queries: Union[str, List[str]], 
               fields: Optional[List[str]] = None,
               top_k: int = SearchConfig.DEFAULT_TOP_K) -> List[Tuple[int, float]]:
        """
        Perform keyword search using Tantivy.
        
        Args:
            queries: Single query string or list of queries
            fields: Fields to search in (None = search all fields)
            top_k: Number of results to return
        
        Returns:
            List of (doc_id, score) tuples
        """
        if isinstance(queries, str):
            queries = [queries]
        
        all_results = {}
        
        for query_str in queries:
            # Clean query
            query_str = clean_text(query_str)
            if not query_str:
                continue
            
            # Build query
            if fields:
                # Search specific fields
                query_parts = []
                for field in fields:
                    field_name = field.replace(" ", "_").lower()
                    query_parts.append(f'{field_name}:"{query_str}"')
                final_query = " OR ".join(query_parts)
            else:
                # Search combined field for cross-field search
                final_query = f'_combined:"{query_str}"'
            
            # Parse and execute query
            try:
                query = self.index.parse_query(final_query)
                results = self.searcher.search(query, top_k)
                
                # Process results
                for score, doc_address in results.hits:
                    doc = self.searcher.doc(doc_address)
                    doc_id = doc.get("doc_id")[0]  # Get document ID
                    
                    # Aggregate scores if document appears multiple times
                    if doc_id not in all_results:
                        all_results[doc_id] = 0
                    all_results[doc_id] += score
                    
            except Exception as e:
                logger.warning(f"Query failed for '{query_str}': {e}")
                continue
        
        # Sort by score and return top_k
        sorted_results = sorted(
            all_results.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        return sorted_results
    
    def get_document(self, doc_id: int) -> Dict[str, Any]:
        """Retrieve full document by ID."""
        query = self.index.parse_query(f"doc_id:{doc_id}")
        results = self.searcher.search(query, 1)
        
        if results.hits:
            _, doc_address = results.hits[0]
            doc = self.searcher.doc(doc_address)
            return {field: doc.get(field) for field in doc}
        return {}


class FAISSSearcher:
    """FAISS-based semantic search engine."""
    
    def __init__(self, index_path: str = SearchConfig.FAISS_INDEX_PATH):
        self.index_path = Path(index_path)
        self.index = None
        self.id_map = {}
        self.embedding_client = None
        self._load_index()
    
    def _load_index(self):
        """Load existing FAISS index."""
        index_file = self.index_path / "index.faiss"
        if not index_file.exists():
            raise ValueError(f"FAISS index not found at {index_file}")
        
        # Load index
        self.index = faiss.read_index(str(index_file))
        
        # Load ID map
        id_map_file = self.index_path / "id_map.json"
        if id_map_file.exists():
            with open(id_map_file, 'r') as f:
                self.id_map = {int(k): v for k, v in json.load(f).items()}
        
        logger.info(f"Loaded FAISS index from {self.index_path}")
    
    async def initialize_embedding_client(self):
        """Initialize embedding client for query encoding."""
        if not self.embedding_client:
            self.embedding_client = OptimizedEmbeddingClient()
    
    async def search(self,
                    query: str,
                    top_k: int = SearchConfig.DEFAULT_TOP_K) -> List[Tuple[int, float]]:
        """
        Perform semantic search using FAISS.
        
        Args:
            query: Query string
            top_k: Number of results to return
        
        Returns:
            List of (doc_id, distance) tuples
        """
        # Initialize embedding client if needed
        await self.initialize_embedding_client()
        
        # Clean query
        query = clean_text(query)
        if not query:
            return []
        
        # Generate query embedding
        query_embedding = await self.embedding_client.get_single_embedding(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Convert to doc IDs
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1 and idx in self.id_map:
                doc_id = self.id_map[idx]
                # Convert L2 distance to similarity score (inverse)
                score = 1.0 / (1.0 + dist)
                results.append((doc_id, score))
        
        return results


class SearchEngine:
    """Main search engine combining keyword and semantic search."""
    
    def __init__(self):
        self.tantivy_searcher = None
        self.faiss_searcher = None
        self.rrf_scorer = RRFScorer()
        self.df = None
        self._initialize()
    
    def _initialize(self):
        """Initialize search components."""
        try:
            self.tantivy_searcher = TantivySearcher()
        except Exception as e:
            logger.warning(f"Failed to initialize Tantivy searcher: {e}")
        
        try:
            self.faiss_searcher = FAISSSearcher()
        except Exception as e:
            logger.warning(f"Failed to initialize FAISS searcher: {e}")
    
    def load_data(self, data_path: str = SearchConfig.DATA_PATH):
        """Load original data for result retrieval."""
        self.df = pd.read_parquet(data_path)
        logger.info(f"Loaded data with {len(self.df)} rows")
    
    def keyword_search(self, 
                      queries: Union[str, List[str]], 
                      top_k: int = SearchConfig.DEFAULT_TOP_K,
                      fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search.
        
        Args:
            queries: Single query or list of queries
            top_k: Number of results to return
            fields: Specific fields to search in
        
        Returns:
            List of result dictionaries with document data and scores
        """
        if not self.tantivy_searcher:
            raise ValueError("Tantivy index not available")
        
        start_time = time.time()
        
        # Perform search
        results = self.tantivy_searcher.search(queries, fields, top_k)
        
        # Retrieve full documents
        output = []
        for doc_id, score in results:
            if self.df is not None and doc_id in self.df.index:
                doc_data = self.df.loc[doc_id].to_dict()
                doc_data['_score'] = score
                doc_data['_id'] = doc_id
                doc_data['_search_type'] = 'keyword'
                output.append(doc_data)
        
        elapsed = time.time() - start_time
        logger.info(f"Keyword search completed in {elapsed:.3f}s")
        
        return output
    
    async def semantic_search(self,
                             query: str,
                             top_k: int = SearchConfig.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: Query string
            top_k: Number of results to return
        
        Returns:
            List of result dictionaries with document data and scores
        """
        if not self.faiss_searcher:
            raise ValueError("FAISS index not available")
        
        start_time = time.time()
        
        # Perform search
        results = await self.faiss_searcher.search(query, top_k)
        
        # Retrieve full documents
        output = []
        for doc_id, score in results:
            if self.df is not None and doc_id in self.df.index:
                doc_data = self.df.loc[doc_id].to_dict()
                doc_data['_score'] = score
                doc_data['_id'] = doc_id
                doc_data['_search_type'] = 'semantic'
                output.append(doc_data)
        
        elapsed = time.time() - start_time
        logger.info(f"Semantic search completed in {elapsed:.3f}s")
        
        return output
    
    async def hybrid_search(self,
                           queries: Union[str, List[str]],
                           top_k: int = SearchConfig.DEFAULT_TOP_K,
                           keyword_weight: float = 1.0,
                           semantic_weight: float = 1.0) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining keyword and semantic search with RRF.
        
        Args:
            queries: Query string or list of queries
            top_k: Number of results to return
            keyword_weight: Weight for keyword search results
            semantic_weight: Weight for semantic search results
        
        Returns:
            List of result dictionaries with document data and RRF scores
        """
        if not self.tantivy_searcher and not self.faiss_searcher:
            raise ValueError("No search indices available")
        
        start_time = time.time()
        
        # Ensure queries is a list
        if isinstance(queries, str):
            queries = [queries]
        
        all_rankings = []
        
        # Perform keyword search if available
        if self.tantivy_searcher:
            keyword_results = self.tantivy_searcher.search(queries, None, top_k * 2)
            if keyword_weight != 1.0:
                # Apply weight
                keyword_results = [(doc_id, score * keyword_weight) 
                                  for doc_id, score in keyword_results]
            all_rankings.append(keyword_results)
        
        # Perform semantic search if available
        if self.faiss_searcher:
            # Use first query for semantic search
            semantic_results = await self.faiss_searcher.search(queries[0], top_k * 2)
            if semantic_weight != 1.0:
                # Apply weight
                semantic_results = [(doc_id, score * semantic_weight) 
                                   for doc_id, score in semantic_results]
            all_rankings.append(semantic_results)
        
        # Apply RRF to combine results
        if len(all_rankings) > 1:
            combined_results = self.rrf_scorer.score(all_rankings)
        else:
            combined_results = all_rankings[0] if all_rankings else []
        
        # Retrieve full documents for top_k results
        output = []
        for doc_id, rrf_score in combined_results[:top_k]:
            if self.df is not None and doc_id in self.df.index:
                doc_data = self.df.loc[doc_id].to_dict()
                doc_data['_score'] = rrf_score
                doc_data['_id'] = doc_id
                doc_data['_search_type'] = 'hybrid'
                output.append(doc_data)
        
        elapsed = time.time() - start_time
        logger.info(f"Hybrid search completed in {elapsed:.3f}s")
        
        # Check if we met the performance target
        if elapsed > SearchConfig.SEARCH_TIMEOUT:
            logger.warning(f"Search exceeded target time: {elapsed:.3f}s > {SearchConfig.SEARCH_TIMEOUT}s")
        
        return output
    
    async def search(self,
                    query: Union[str, List[str]],
                    mode: str = "hybrid",
                    top_k: int = SearchConfig.DEFAULT_TOP_K,
                    **kwargs) -> List[Dict[str, Any]]:
        """
        Unified search interface.
        
        Args:
            query: Query string or list of queries
            mode: Search mode - "keyword", "semantic", or "hybrid"
            top_k: Number of results to return
            **kwargs: Additional arguments for specific search modes
        
        Returns:
            List of result dictionaries
        """
        if mode == "keyword":
            return self.keyword_search(query, top_k, **kwargs)
        elif mode == "semantic":
            if isinstance(query, list):
                query = query[0]  # Semantic search uses single query
            return await self.semantic_search(query, top_k)
        elif mode == "hybrid":
            return await self.hybrid_search(query, top_k, **kwargs)
        else:
            raise ValueError(f"Invalid search mode: {mode}. Use 'keyword', 'semantic', or 'hybrid'")


# Singleton instance
_search_engine = None

def get_search_engine() -> SearchEngine:
    """Get or create search engine singleton."""
    global _search_engine
    if _search_engine is None:
        _search_engine = SearchEngine()
        _search_engine.load_data()
    return _search_engine