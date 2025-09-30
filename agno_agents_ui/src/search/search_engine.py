"""Main search engine implementation with all search modes."""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.models import SearchMode, SearchRequest, SearchResult
from src.core.config import SearchConfig
from src.indexers.duckdb_indexer import DuckDBIndexer
from src.indexers.tantivy_indexer import TantivyIndexer
from src.indexers.faiss_indexer import FAISSIndexer
from src.utils.rrf import merge_search_results


class SearchEngine:
    """Main search engine with support for all search modes."""

    def __init__(self, config: SearchConfig):
        """Initialize the search engine.

        Args:
            config: Search configuration
        """
        self.config = config

        # Initialize indexers
        self.duckdb_indexer = DuckDBIndexer(config.index_path)
        self.tantivy_indexer = TantivyIndexer(config.index_path)
        self.faiss_indexer = FAISSIndexer(config.index_path, config.vector_dimension)

        # Load indexes
        self._load_indexes()

    def _load_indexes(self):
        """Load all indexes."""
        print("Loading search indexes...")

        # Check if indexes exist
        if not self.duckdb_indexer.index_exists():
            raise ValueError(f"DuckDB index not found at {self.config.index_path}")

        if not self.tantivy_indexer.index_exists():
            raise ValueError(f"Tantivy index not found at {self.config.index_path}")

        if not self.faiss_indexer.index_exists():
            raise ValueError(f"FAISS index not found at {self.config.index_path}")

        # Load Tantivy and FAISS indexes
        self.tantivy_indexer.load_index()
        self.faiss_indexer.load_index()

        print("✓ All indexes loaded successfully")

    def search_dataframe(self, column_value_pairs: Dict[str, Any]) -> tuple[List[Dict], int]:
        """Search using dataframe mode (column-value filtering).

        Args:
            column_value_pairs: Dictionary of column-value pairs

        Returns:
            Tuple of (results, count)
        """
        return self.duckdb_indexer.search(column_value_pairs)

    def search_keywords(self, keywords: List[str], top_k: int = 5) -> List[tuple[int, float, Dict[str, Any]]]:
        """Search using keywords across all text columns.

        Args:
            keywords: List of keywords to search
            top_k: Number of top results

        Returns:
            List of (doc_id, score, document) tuples
        """
        # Get all text columns for searching
        text_columns = ["title", "description", "content", "abstract", "summary",
                       "body", "metadata", "tags", "keywords", "comments", "notes"]

        # Search in each column and collect results
        all_results = self.tantivy_indexer.search_in_columns(keywords, text_columns, limit=100)

        # Group results by document ID and aggregate scores
        doc_scores = {}
        doc_map = {}

        for doc_id, score, document in all_results:
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0
                doc_map[doc_id] = document
            doc_scores[doc_id] += score

        # Sort by aggregated score and return top K
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        results = []

        for doc_id, total_score in sorted_docs[:top_k]:
            results.append((doc_id, total_score, doc_map[doc_id]))

        return results

    def search_vector(self, vector: List[float], top_k: int = 5) -> List[tuple[int, float, Dict[str, Any]]]:
        """Search using vector similarity across embedding columns.

        Args:
            vector: Query vector
            top_k: Number of top results

        Returns:
            List of (doc_id, score, document) tuples
        """
        # Get embedding columns from FAISS indexer
        embedding_columns = list(self.faiss_indexer.indexes.keys())

        # Search across all embedding columns
        all_results = self.faiss_indexer.search_multiple_columns(
            np.array(vector),
            embedding_columns,
            k=100
        )

        # Group results by document ID and take max score
        doc_scores = {}
        doc_map = {}

        for doc_id, score, document in all_results:
            if doc_id not in doc_scores:
                doc_scores[doc_id] = score
                doc_map[doc_id] = document
            else:
                doc_scores[doc_id] = max(doc_scores[doc_id], score)

        # Sort by score and return top K
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        results = []

        for doc_id, score in sorted_docs[:top_k]:
            results.append((doc_id, score, doc_map[doc_id]))

        return results

    def search_hybrid(
        self,
        column_value_pairs: Optional[Dict[str, Any]] = None,
        keywords: Optional[List[str]] = None,
        vector: Optional[List[float]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining multiple search modes.

        Args:
            column_value_pairs: Optional column-value pairs for filtering
            keywords: Optional keywords for text search
            vector: Optional vector for similarity search
            top_k: Number of top results

        Returns:
            List of merged results
        """
        results_to_merge = []

        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []

            # Submit dataframe search
            if column_value_pairs:
                future = executor.submit(self.search_dataframe, column_value_pairs)
                futures.append(('dataframe', future))

            # Submit keyword search
            if keywords:
                future = executor.submit(self.search_keywords, keywords, 100)
                futures.append(('keyword', future))

            # Submit vector search
            if vector:
                future = executor.submit(self.search_vector, vector, 100)
                futures.append(('vector', future))

            # Collect results
            for search_type, future in futures:
                try:
                    result = future.result(timeout=5)
                    if search_type == 'dataframe':
                        # Dataframe returns (results, count)
                        results_to_merge.append(('dataframe', result[0][:100]))
                    else:
                        results_to_merge.append((search_type, result))
                except Exception as e:
                    print(f"Error in {search_type} search: {e}")

        # Merge results using RRF
        dataframe_results = None
        keyword_results = None
        vector_results = None

        for search_type, results in results_to_merge:
            if search_type == 'dataframe':
                dataframe_results = results
            elif search_type == 'keyword':
                keyword_results = results
            elif search_type == 'vector':
                vector_results = results

        return merge_search_results(
            dataframe_results=dataframe_results,
            keyword_results=keyword_results,
            vector_results=vector_results,
            top_n=top_k
        )

    def search(self, request: SearchRequest) -> SearchResult:
        """Main search method handling all search modes.

        Args:
            request: Search request with mode and parameters

        Returns:
            Search result with status, count, results, and timing
        """
        start_time = time.time()
        results = []
        count = 0
        status = "failed"

        try:
            if request.mode == SearchMode.DATAFRAME:
                # Dataframe search
                if not request.column_value_pairs:
                    raise ValueError("column_value_pairs required for dataframe mode")

                results, count = self.search_dataframe(request.column_value_pairs)
                status = "success" if count > 0 else "failed"

            elif request.mode == SearchMode.KEYWORD:
                # Keyword search
                if not request.keywords:
                    raise ValueError("keywords required for keyword mode")

                keyword_results = self.search_keywords(request.keywords, request.top_k)
                results = [doc for _, _, doc in keyword_results]
                count = len(results)
                status = "success" if count > 0 else "failed"

            elif request.mode == SearchMode.VECTOR:
                # Vector search
                if not request.vector:
                    raise ValueError("vector required for vector mode")

                vector_results = self.search_vector(request.vector, request.top_k)
                results = [doc for _, _, doc in vector_results]
                count = len(results)
                status = "success" if count > 0 else "failed"

            elif request.mode == SearchMode.HYBRID:
                # Hybrid search
                if not any([request.column_value_pairs, request.keywords, request.vector]):
                    raise ValueError("At least one search parameter required for hybrid mode")

                results = self.search_hybrid(
                    column_value_pairs=request.column_value_pairs,
                    keywords=request.keywords,
                    vector=request.vector,
                    top_k=request.top_k
                )
                count = len(results)
                status = "success" if count > 0 else "failed"

            else:
                raise ValueError(f"Unknown search mode: {request.mode}")

        except Exception as e:
            print(f"Search error: {e}")
            status = "failed"
            results = []
            count = 0

        # Calculate time taken
        time_taken = time.time() - start_time

        return SearchResult(
            status=status,
            count=count,
            results=results,
            time_taken=time_taken
        )