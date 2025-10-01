"""
Advanced search engine with multiple modes: dataframe, keyword, vector, and hybrid.
Optimized for production with <200ms response time for 95% of queries.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import duckdb
import tantivy
import faiss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SearchEngine:
    def __init__(self, index_path: str):
        """
        Initialize the search engine with pre-built indices.

        Args:
            index_path: Path to the index directory
        """
        self.index_path = Path(index_path)

        # Paths to different index components
        self.duckdb_path = self.index_path / "duckdb_index.db"
        self.tantivy_path = self.index_path / "tantivy_index"
        self.faiss_path = self.index_path / "faiss_indices"
        self.metadata_path = self.index_path / "metadata.json"

        # Load metadata
        self._load_metadata()

        # Initialize connections
        self.conn = None
        self.tantivy_index = None
        self.faiss_indices = {}

        # Initialize indices
        self._initialize_indices()

    def _load_metadata(self):
        """Load metadata from the index."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")

        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.total_rows = self.metadata["total_rows"]
        self.columns = self.metadata["columns"]
        self.embedding_columns = self.metadata["embedding_columns"]
        self.vector_dimension = self.metadata["vector_dimension"]

    def _initialize_indices(self):
        """Initialize all indices for searching."""
        # Initialize DuckDB connection
        if not self.duckdb_path.exists():
            raise FileNotFoundError(f"DuckDB index not found at {self.duckdb_path}")

        self.conn = duckdb.connect(str(self.duckdb_path), read_only=True)

        # Initialize Tantivy index
        if not self.tantivy_path.exists():
            raise FileNotFoundError(f"Tantivy index not found at {self.tantivy_path}")

        self.tantivy_index = tantivy.Index.open(str(self.tantivy_path))

        # Initialize FAISS indices
        for col in self.embedding_columns:
            index_file = self.faiss_path / f"{col}.index"
            if index_file.exists():
                self.faiss_indices[col] = faiss.read_index(str(index_file))

    def _reciprocal_rank_fusion(self, results_lists: List[List[int]], k: int = 60) -> List[int]:
        """
        Perform Reciprocal Rank Fusion (RRF) to combine multiple ranked lists.

        Args:
            results_lists: List of ranked document ID lists
            k: RRF parameter (default 60)

        Returns:
            Fused list of document IDs
        """
        scores = {}

        for result_list in results_lists:
            for rank, doc_id in enumerate(result_list):
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] += 1.0 / (k + rank + 1)

        # Sort by score and return top results
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs]

    def search_dataframe(
        self,
        column_value_pairs: Dict[str, Any]
    ) -> Tuple[List[Dict], float]:
        """
        Search using SQL-like queries on structured data.

        Args:
            column_value_pairs: Dictionary of column-value pairs to filter

        Returns:
            Tuple of (results, time_taken)
        """
        start_time = time.time()

        # Build WHERE clause
        where_conditions = []
        for col, val in column_value_pairs.items():
            if col not in self.columns:
                continue

            if val is None or (isinstance(val, str) and val.lower() in ['null', 'none', 'nan']):
                where_conditions.append(f"{col} IS NULL")
            elif isinstance(val, str):
                # Escape single quotes in string values
                val_escaped = val.replace("'", "''")
                where_conditions.append(f"{col} = '{val_escaped}'")
            elif isinstance(val, bool):
                where_conditions.append(f"{col} = {val}")
            elif isinstance(val, (int, float)):
                where_conditions.append(f"{col} = {val}")
            else:
                where_conditions.append(f"CAST({col} AS VARCHAR) = '{str(val)}'")

        if not where_conditions:
            return [], time.time() - start_time

        where_clause = " AND ".join(where_conditions)
        query = f"SELECT * FROM documents WHERE {where_clause}"

        try:
            # Execute query
            result_df = self.conn.execute(query).fetchdf()

            # Convert to list of dictionaries
            results = result_df.to_dict('records')

            time_taken = time.time() - start_time
            return results, time_taken

        except Exception as e:
            print(f"Error in dataframe search: {e}")
            return [], time.time() - start_time

    def search_keyword(
        self,
        keywords: List[str],
        top_k: int = 5
    ) -> Tuple[List[Dict], float]:
        """
        Search using full-text search with Tantivy.

        Args:
            keywords: List of keywords to search
            top_k: Number of top results to return

        Returns:
            Tuple of (results, time_taken)
        """
        start_time = time.time()

        searcher = self.tantivy_index.searcher()

        # Get all text field names - we'll use the columns from metadata
        # that are text columns
        text_fields = [col for col in self.columns if col not in ['id', 'quantity', 'price', 'rating', 'views', 'clicks',
                                                                   'temperature', 'humidity', 'pressure', 'weight', 'height',
                                                                   'width', 'depth', 'discount_percentage', 'tax_rate',
                                                                   'profit_margin', 'is_active', 'is_verified', 'has_discount']]

        all_doc_scores = {}

        # Search each keyword in each field
        for keyword in keywords:
            for field in text_fields:
                try:
                    # Create query - parse_query needs field list
                    query = self.tantivy_index.parse_query(keyword, [field])

                    # Search
                    search_results = searcher.search(query, limit=top_k * 2)

                    # Collect results with scores
                    for score, doc_address in search_results.hits:
                        doc = searcher.doc(doc_address)
                        doc_id = doc.get_first("doc_id")

                        if doc_id not in all_doc_scores:
                            all_doc_scores[doc_id] = []
                        all_doc_scores[doc_id].append(score)

                except Exception as e:
                    continue

        # Aggregate scores and rank
        doc_rankings = []
        for field in text_fields:
            field_docs = []
            for doc_id, scores in all_doc_scores.items():
                if scores:
                    field_docs.append((doc_id, sum(scores)))

            # Sort by score
            field_docs.sort(key=lambda x: x[1], reverse=True)
            doc_rankings.append([doc_id for doc_id, _ in field_docs[:top_k * 2]])

        # Apply RRF to merge rankings
        if doc_rankings:
            final_doc_ids = self._reciprocal_rank_fusion(doc_rankings)[:top_k]
        else:
            final_doc_ids = []

        # Fetch full documents from DuckDB
        results = []
        if final_doc_ids:
            # DuckDB uses 0-based row indices, so we can use OFFSET/LIMIT for each doc
            for doc_id in final_doc_ids[:top_k]:
                try:
                    query = f"SELECT * FROM documents LIMIT 1 OFFSET {doc_id}"
                    result = self.conn.execute(query).fetchdf()
                    if not result.empty:
                        results.append(result.iloc[0].to_dict())
                except Exception as e:
                    continue

        time_taken = time.time() - start_time
        return results[:top_k], time_taken

    def search_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> Tuple[List[Dict], float]:
        """
        Search using vector similarity with FAISS.

        Args:
            query_vector: Query vector of appropriate dimension
            top_k: Number of top results to return

        Returns:
            Tuple of (results, time_taken)
        """
        start_time = time.time()

        if not self.faiss_indices:
            return [], time.time() - start_time

        # Ensure query vector is the right shape and type
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)

        if query_vector.shape[1] != self.vector_dimension:
            print(f"Error: Query vector dimension {query_vector.shape[1]} doesn't match index dimension {self.vector_dimension}")
            return [], time.time() - start_time

        all_results = []

        # Search in each embedding column
        for col, index in self.faiss_indices.items():
            try:
                # Search in FAISS index
                distances, indices = index.search(query_vector, top_k * 2)

                # Convert to list of (doc_id, distance) tuples
                col_results = []
                for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx != -1:  # Valid result
                        col_results.append(int(idx))

                all_results.append(col_results)

            except Exception as e:
                print(f"Error searching in column {col}: {e}")
                continue

        # Apply RRF to merge results from different columns
        if all_results:
            final_doc_ids = self._reciprocal_rank_fusion(all_results)[:top_k]
        else:
            final_doc_ids = []

        # Fetch full documents from DuckDB
        results = []
        if final_doc_ids:
            # DuckDB uses 0-based row indices, so we can use OFFSET/LIMIT for each doc
            for doc_id in final_doc_ids[:top_k]:
                try:
                    query = f"SELECT * FROM documents LIMIT 1 OFFSET {doc_id}"
                    result = self.conn.execute(query).fetchdf()
                    if not result.empty:
                        results.append(result.iloc[0].to_dict())
                except Exception as e:
                    continue

        time_taken = time.time() - start_time
        return results[:top_k], time_taken

    def search_hybrid(
        self,
        column_value_pairs: Optional[Dict[str, Any]] = None,
        keywords: Optional[List[str]] = None,
        query_vector: Optional[np.ndarray] = None,
        top_k: int = 5
    ) -> Tuple[List[Dict], float]:
        """
        Hybrid search combining multiple search modes in parallel.

        Args:
            column_value_pairs: Dictionary for dataframe search
            keywords: List of keywords for text search
            query_vector: Vector for similarity search
            top_k: Number of top results to return

        Returns:
            Tuple of (results, time_taken)
        """
        start_time = time.time()

        search_tasks = []
        search_results = {}

        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit dataframe search if column_value_pairs provided
            if column_value_pairs:
                future = executor.submit(self.search_dataframe, column_value_pairs)
                search_tasks.append(('dataframe', future))

            # Submit keyword search if keywords provided
            if keywords:
                future = executor.submit(self.search_keyword, keywords, top_k * 2)
                search_tasks.append(('keyword', future))

            # Submit vector search if query_vector provided
            if query_vector is not None:
                future = executor.submit(self.search_vector, query_vector, top_k * 2)
                search_tasks.append(('vector', future))

            # Collect results
            for search_type, future in search_tasks:
                try:
                    results, _ = future.result(timeout=1.0)
                    search_results[search_type] = results
                except Exception as e:
                    print(f"Error in {search_type} search: {e}")
                    search_results[search_type] = []

        # Combine results using RRF
        all_doc_ids_lists = []

        # For dataframe results, use all of them (no top_k limit)
        if 'dataframe' in search_results and search_results['dataframe']:
            df_doc_ids = list(range(len(search_results['dataframe'])))
            all_doc_ids_lists.append(df_doc_ids)

        # For keyword and vector results, extract doc IDs
        for search_type in ['keyword', 'vector']:
            if search_type in search_results and search_results[search_type]:
                # Extract row indices or IDs from results
                doc_ids = []
                for i, result in enumerate(search_results[search_type]):
                    # Use the index position as doc_id
                    doc_ids.append(i)
                all_doc_ids_lists.append(doc_ids)

        # Apply RRF if we have multiple result lists
        if len(all_doc_ids_lists) > 1:
            final_indices = self._reciprocal_rank_fusion(all_doc_ids_lists)[:top_k]

            # Collect final results
            final_results = []
            result_map = {}

            # Build result map
            for search_type, results in search_results.items():
                for i, result in enumerate(results):
                    result_map[i] = result

            # Get results in RRF order
            for idx in final_indices:
                if idx in result_map:
                    final_results.append(result_map[idx])

        elif len(all_doc_ids_lists) == 1:
            # If only one search mode was used, return its results
            for results in search_results.values():
                if results:
                    final_results = results[:top_k]
                    break
            else:
                final_results = []
        else:
            final_results = []

        time_taken = time.time() - start_time
        return final_results, time_taken

    def search(
        self,
        search_mode: str,
        column_value_pairs: Optional[Dict[str, Any]] = None,
        keywords: Optional[List[str]] = None,
        query_vector: Optional[np.ndarray] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Main search interface supporting multiple modes.

        Args:
            search_mode: One of 'dataframe', 'keyword', 'vector', 'hybrid'
            column_value_pairs: For dataframe mode
            keywords: For keyword mode
            query_vector: For vector mode
            top_k: Number of results (not applicable for dataframe mode)

        Returns:
            Dictionary with status, count, results, and time_taken
        """
        results = []
        time_taken = 0.0

        try:
            if search_mode == 'dataframe':
                if not column_value_pairs:
                    return {
                        "status": "failed",
                        "count": 0,
                        "results": [],
                        "time_taken": 0.0,
                        "error": "column_value_pairs required for dataframe mode"
                    }
                results, time_taken = self.search_dataframe(column_value_pairs)

            elif search_mode == 'keyword':
                if not keywords:
                    return {
                        "status": "failed",
                        "count": 0,
                        "results": [],
                        "time_taken": 0.0,
                        "error": "keywords required for keyword mode"
                    }
                results, time_taken = self.search_keyword(keywords, top_k)

            elif search_mode == 'vector':
                if query_vector is None:
                    return {
                        "status": "failed",
                        "count": 0,
                        "results": [],
                        "time_taken": 0.0,
                        "error": "query_vector required for vector mode"
                    }
                results, time_taken = self.search_vector(query_vector, top_k)

            elif search_mode == 'hybrid':
                if not any([column_value_pairs, keywords, query_vector is not None]):
                    return {
                        "status": "failed",
                        "count": 0,
                        "results": [],
                        "time_taken": 0.0,
                        "error": "At least one search parameter required for hybrid mode"
                    }
                results, time_taken = self.search_hybrid(
                    column_value_pairs, keywords, query_vector, top_k
                )

            else:
                return {
                    "status": "failed",
                    "count": 0,
                    "results": [],
                    "time_taken": 0.0,
                    "error": f"Invalid search mode: {search_mode}"
                }

            # Prepare response
            count = len(results)
            status = "success" if count > 0 else "failed"

            return {
                "status": status,
                "count": count,
                "results": results,
                "time_taken": time_taken
            }

        except Exception as e:
            return {
                "status": "failed",
                "count": 0,
                "results": [],
                "time_taken": time_time() - start_time if 'start_time' in locals() else 0.0,
                "error": str(e)
            }

def main():
    """Main function for testing the search engine."""
    # Load configuration
    load_dotenv()

    index_path = os.getenv("INDEX_PATH", "./indices")

    # Initialize search engine
    print("Initializing search engine...")
    engine = SearchEngine(index_path)

    print("\nSearch engine ready!")
    print(f"Total documents: {engine.total_rows:,}")
    print(f"Vector dimensions: {engine.vector_dimension}")
    print(f"Embedding columns: {engine.embedding_columns}")

    # Example searches
    print("\n" + "="*50)
    print("Running example searches...")
    print("="*50)

    # 1. Dataframe search
    print("\n1. Dataframe Search:")
    result = engine.search(
        search_mode="dataframe",
        column_value_pairs={"status": "active", "type": "type_a"}
    )
    print(f"   Status: {result['status']}")
    print(f"   Count: {result['count']}")
    print(f"   Time: {result['time_taken']*1000:.2f}ms")

    # 2. Keyword search
    print("\n2. Keyword Search:")
    result = engine.search(
        search_mode="keyword",
        keywords=["search", "engine", "data"]
    )
    print(f"   Status: {result['status']}")
    print(f"   Count: {result['count']}")
    print(f"   Time: {result['time_taken']*1000:.2f}ms")

    # 3. Vector search
    print("\n3. Vector Search:")
    random_vector = np.random.randn(engine.vector_dimension).astype(np.float32)
    result = engine.search(
        search_mode="vector",
        query_vector=random_vector
    )
    print(f"   Status: {result['status']}")
    print(f"   Count: {result['count']}")
    print(f"   Time: {result['time_taken']*1000:.2f}ms")

    # 4. Hybrid search
    print("\n4. Hybrid Search:")
    result = engine.search(
        search_mode="hybrid",
        column_value_pairs={"is_active": True},
        keywords=["machine", "learning"],
        query_vector=random_vector
    )
    print(f"   Status: {result['status']}")
    print(f"   Count: {result['count']}")
    print(f"   Time: {result['time_taken']*1000:.2f}ms")

if __name__ == "__main__":
    main()