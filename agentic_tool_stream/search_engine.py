"""
High-Performance Search Engine
Supports dataframe, keyword, vector, and hybrid search modes
"""

import os
import json
import time
import duckdb
import faiss
import tantivy
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from embeddings_client import EmbeddingsClient
import threading

class SearchEngine:
    def __init__(
        self,
        index_path: str = "indices",
        max_results: int = 5,
        max_workers: int = 4
    ):
        self.index_path = index_path
        self.max_results = max_results
        self.max_workers = max_workers

        # Load metadata
        self.metadata = self._load_metadata()

        # Initialize connections
        self.duckdb_conn = None
        self.tantivy_index = None
        self.faiss_indices = {}
        self.text_columns = []

        # Thread lock for DuckDB connection
        self.db_lock = threading.Lock()

        # Initialize components
        self._initialize_components()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from disk"""
        metadata_path = os.path.join(self.index_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        raise FileNotFoundError(f"Metadata not found at {metadata_path}. Run ingestion first.")

    def _initialize_components(self):
        """Initialize search components"""
        # Initialize DuckDB connection
        duckdb_path = os.path.join(self.index_path, "duckdb.db")
        if os.path.exists(duckdb_path):
            self.duckdb_conn = duckdb.connect(duckdb_path, read_only=True)
        else:
            raise FileNotFoundError(f"DuckDB index not found at {duckdb_path}")

        # Initialize Tantivy index
        tantivy_path = os.path.join(self.index_path, "tantivy")
        if os.path.exists(tantivy_path):
            # Load schema
            try:
                # Load text columns
                text_columns_path = os.path.join(tantivy_path, "text_columns.json")
                if os.path.exists(text_columns_path):
                    with open(text_columns_path, 'r') as f:
                        self.text_columns = json.load(f)

                # Open existing index
                self.tantivy_index = tantivy.Index.open(tantivy_path)
            except Exception as e:
                print(f"Warning: Could not load Tantivy index: {e}")

        # Initialize FAISS indices
        faiss_path = os.path.join(self.index_path, "faiss")
        if os.path.exists(faiss_path):
            faiss_metadata_path = os.path.join(faiss_path, "metadata.json")
            if os.path.exists(faiss_metadata_path):
                with open(faiss_metadata_path, 'r') as f:
                    faiss_metadata = json.load(f)

                for col_name in faiss_metadata.keys():
                    index_path = os.path.join(faiss_path, f"{col_name}.index")
                    if os.path.exists(index_path):
                        self.faiss_indices[col_name] = faiss.read_index(index_path)

    def _reciprocal_rank_fusion(
        self,
        results_lists: List[List[Dict]],
        k: int = 60
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion to combine multiple result lists
        RRF score = sum(1 / (k + rank))
        """
        # Combine all results with RRF scoring
        rrf_scores = {}
        result_data = {}

        for results in results_lists:
            for rank, result in enumerate(results):
                # Use ID or index as key
                if 'id' in result:
                    key = result['id']
                elif 'index' in result:
                    key = result['index']
                else:
                    key = str(result)

                # Calculate RRF score
                if key not in rrf_scores:
                    rrf_scores[key] = 0
                    result_data[key] = result

                rrf_scores[key] += 1.0 / (k + rank + 1)

        # Sort by RRF score
        sorted_results = sorted(
            [(score, key) for key, score in rrf_scores.items()],
            reverse=True
        )

        # Return top results
        final_results = []
        for score, key in sorted_results[:self.max_results]:
            result = result_data[key].copy()
            result['rrf_score'] = score
            final_results.append(result)

        return final_results

    def search_dataframe(
        self,
        column_value_pairs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Search using exact column-value matching
        """
        start_time = time.time()

        # Build SQL query
        conditions = []
        params = []

        for col, val in column_value_pairs.items():
            if val is None:
                conditions.append(f"{col} IS NULL")
            else:
                conditions.append(f"{col} = ?")
                params.append(val)

        where_clause = " AND ".join(conditions)
        query = f"SELECT * FROM search_data WHERE {where_clause}"

        # Execute query
        with self.db_lock:
            if params:
                result = self.duckdb_conn.execute(query, params).fetchall()
            else:
                result = self.duckdb_conn.execute(query).fetchall()

            # Get column names
            columns = [desc[0] for desc in self.duckdb_conn.description]

        # Convert to list of dicts
        results = [dict(zip(columns, row)) for row in result]

        elapsed_time = time.time() - start_time

        return {
            "status": "success" if len(results) > 0 else "failed",
            "count": len(results),
            "results": results,  # Return all matching rows for dataframe mode
            "time_taken": f"{elapsed_time:.3f}s",
            "mode": "dataframe"
        }

    def search_keyword(
        self,
        keywords: List[str]
    ) -> Dict[str, Any]:
        """
        Full-text keyword search using Tantivy
        """
        start_time = time.time()

        if not self.tantivy_index:
            return {
                "status": "failed",
                "count": 0,
                "results": [],
                "time_taken": f"{time.time() - start_time:.3f}s",
                "mode": "keyword",
                "error": "Tantivy index not available"
            }

        # Search for each keyword
        all_results = []
        searcher = self.tantivy_index.searcher()

        for keyword in keywords:
            # Build query for all text fields
            query_parts = []
            for col in self.text_columns:
                query_parts.append(f'{col}:"{keyword}"')

            query_str = " OR ".join(query_parts)

            try:
                # Parse and execute query
                query = self.tantivy_index.parse_query(query_str, self.text_columns)
                search_result = searcher.search(query, limit=100)

                # Get document IDs
                doc_results = []
                for score, doc_id in search_result.hits:
                    doc = searcher.doc(doc_id)
                    doc_dict = {"score": score, "doc_id": doc["doc_id"][0]}
                    doc_results.append(doc_dict)

                all_results.append(doc_results)
            except Exception as e:
                print(f"Warning: Search error for keyword '{keyword}': {e}")

        # Use RRF to merge results
        if all_results:
            # Convert to format expected by RRF
            merged = self._reciprocal_rank_fusion(all_results)

            # Fetch actual data from DuckDB
            if merged:
                doc_ids = [r['doc_id'] for r in merged]
                placeholders = ','.join(['?' for _ in doc_ids])
                query = f"SELECT * FROM search_data WHERE rowid IN ({placeholders})"

                with self.db_lock:
                    result = self.duckdb_conn.execute(query, doc_ids).fetchall()
                    columns = [desc[0] for desc in self.duckdb_conn.description]

                results = [dict(zip(columns, row)) for row in result]
            else:
                results = []
        else:
            results = []

        elapsed_time = time.time() - start_time

        return {
            "status": "success" if len(results) > 0 else "failed",
            "count": len(results),
            "results": results[:self.max_results],
            "time_taken": f"{elapsed_time:.3f}s",
            "mode": "keyword"
        }

    def search_vector(
        self,
        query_vector: Union[List[float], np.ndarray]
    ) -> Dict[str, Any]:
        """
        Vector similarity search using FAISS
        """
        start_time = time.time()

        if not self.faiss_indices:
            return {
                "status": "failed",
                "count": 0,
                "results": [],
                "time_taken": f"{time.time() - start_time:.3f}s",
                "mode": "vector",
                "error": "FAISS indices not available"
            }

        # Convert to numpy array
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector, dtype=np.float32)
        else:
            query_vector = query_vector.astype(np.float32)

        # Reshape for FAISS
        query_vector = query_vector.reshape(1, -1)

        # Search in each FAISS index
        all_results = []

        for col_name, index in self.faiss_indices.items():
            try:
                # Search
                distances, indices = index.search(query_vector, k=100)

                # Convert to results format
                col_results = []
                for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx >= 0:  # Valid index
                        col_results.append({
                            "index": int(idx),
                            "distance": float(dist),
                            "column": col_name
                        })

                all_results.append(col_results)
            except Exception as e:
                print(f"Warning: Vector search error for column '{col_name}': {e}")

        # Use RRF to merge results
        if all_results:
            merged = self._reciprocal_rank_fusion(all_results)

            # Fetch actual data from DuckDB
            if merged:
                indices = [r['index'] for r in merged]
                placeholders = ','.join(['?' for _ in indices])
                query = f"SELECT * FROM search_data WHERE rowid IN ({placeholders})"

                with self.db_lock:
                    result = self.duckdb_conn.execute(query, indices).fetchall()
                    columns = [desc[0] for desc in self.duckdb_conn.description]

                results = []
                for row, merge_info in zip(result, merged):
                    result_dict = dict(zip(columns, row))
                    result_dict['vector_distance'] = merge_info.get('distance', 0)
                    result_dict['vector_column'] = merge_info.get('column', '')
                    results.append(result_dict)
            else:
                results = []
        else:
            results = []

        elapsed_time = time.time() - start_time

        return {
            "status": "success" if len(results) > 0 else "failed",
            "count": len(results),
            "results": results[:self.max_results],
            "time_taken": f"{elapsed_time:.3f}s",
            "mode": "vector"
        }

    def search_hybrid(
        self,
        column_value_pairs: Optional[Dict[str, Any]] = None,
        keywords: Optional[List[str]] = None,
        query_vector: Optional[Union[List[float], np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Hybrid search combining all modes with parallel execution
        """
        start_time = time.time()

        # Collect all search tasks
        search_tasks = []
        search_results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            # Submit dataframe search
            if column_value_pairs:
                future = executor.submit(self.search_dataframe, column_value_pairs)
                futures.append(("dataframe", future))

            # Submit keyword search
            if keywords:
                future = executor.submit(self.search_keyword, keywords)
                futures.append(("keyword", future))

            # Submit vector search
            if query_vector is not None:
                future = executor.submit(self.search_vector, query_vector)
                futures.append(("vector", future))

            # Collect results
            mode_results = {}
            for mode, future in futures:
                try:
                    result = future.result(timeout=10)
                    mode_results[mode] = result
                    if result['results']:
                        search_results.append(result['results'])
                except Exception as e:
                    print(f"Warning: {mode} search failed: {e}")

        # Merge results using RRF
        if search_results:
            # Convert results to consistent format for RRF
            formatted_results = []
            for results_list in search_results:
                formatted = []
                for r in results_list:
                    if 'id' in r:
                        formatted.append({"id": r['id'], "data": r})
                    else:
                        # Use first unique field as ID
                        for key in ['email', 'url', 'id']:
                            if key in r:
                                formatted.append({"id": r[key], "data": r})
                                break
                        else:
                            # Fallback to string representation
                            formatted.append({"id": str(r), "data": r})
                formatted_results.append(formatted)

            # Apply RRF
            merged = self._reciprocal_rank_fusion(formatted_results)

            # Extract actual data
            final_results = []
            seen_ids = set()
            for item in merged:
                if 'data' in item:
                    data = item['data']
                    # Add RRF score
                    data['rrf_score'] = item.get('rrf_score', 0)

                    # Deduplicate
                    item_id = item.get('id', str(data))
                    if item_id not in seen_ids:
                        final_results.append(data)
                        seen_ids.add(item_id)
        else:
            final_results = []

        elapsed_time = time.time() - start_time

        return {
            "status": "success" if len(final_results) > 0 else "failed",
            "count": len(final_results),
            "results": final_results[:self.max_results],
            "time_taken": f"{elapsed_time:.3f}s",
            "mode": "hybrid",
            "modes_used": list(mode_results.keys())
        }

    def search(
        self,
        mode: str,
        column_value_pairs: Optional[Dict[str, Any]] = None,
        keywords: Optional[List[str]] = None,
        query_vector: Optional[Union[List[float], np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Main search interface
        """
        if mode == "dataframe":
            if not column_value_pairs:
                return {
                    "status": "failed",
                    "count": 0,
                    "results": [],
                    "time_taken": "0s",
                    "error": "column_value_pairs required for dataframe mode"
                }
            return self.search_dataframe(column_value_pairs)

        elif mode == "keyword":
            if not keywords:
                return {
                    "status": "failed",
                    "count": 0,
                    "results": [],
                    "time_taken": "0s",
                    "error": "keywords required for keyword mode"
                }
            return self.search_keyword(keywords)

        elif mode == "vector":
            if query_vector is None:
                return {
                    "status": "failed",
                    "count": 0,
                    "results": [],
                    "time_taken": "0s",
                    "error": "query_vector required for vector mode"
                }
            return self.search_vector(query_vector)

        elif mode == "hybrid":
            if not any([column_value_pairs, keywords, query_vector is not None]):
                return {
                    "status": "failed",
                    "count": 0,
                    "results": [],
                    "time_taken": "0s",
                    "error": "At least one search parameter required for hybrid mode"
                }
            return self.search_hybrid(column_value_pairs, keywords, query_vector)

        else:
            return {
                "status": "failed",
                "count": 0,
                "results": [],
                "time_taken": "0s",
                "error": f"Invalid search mode: {mode}"
            }

    def close(self):
        """Clean up resources"""
        if self.duckdb_conn:
            self.duckdb_conn.close()