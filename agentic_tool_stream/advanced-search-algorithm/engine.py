"""
Search engine module implementing keyword, semantic, and combined search.
"""
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
import faiss
import tantivy
import pickle
import config
from utils import EmbeddingClient, reciprocal_rank_fusion, handle_edge_cases


class SearchEngine:
    """Fast search engine with keyword, semantic, and combined search capabilities."""

    def __init__(self):
        """Initialize the search engine with pre-built indices."""
        self.tantivy_index_path = Path(config.INDEX_PATH) / "tantivy"
        self.faiss_index_path = Path(config.INDEX_PATH) / "faiss"
        self.metadata_path = Path(config.INDEX_PATH) / "metadata.pkl"
        self.data_path = Path(config.INDEX_PATH) / "data.pkl"

        # Load indices
        self._load_indices()

        # Initialize embedding client for semantic search
        self.embedding_client = EmbeddingClient()

    def _load_indices(self):
        """Load all pre-built indices from disk."""
        # Load Tantivy index
        if not self.tantivy_index_path.exists():
            raise ValueError(f"Tantivy index not found at {self.tantivy_index_path}. Please run ingestion first.")

        schema_builder = tantivy.SchemaBuilder()
        # Recreate schema (must match the one used during indexing)
        schema_builder.add_integer_field("doc_id", indexed=True, stored=True)

        # Load metadata to get columns
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        for col in self.metadata['columns']:
            schema_builder.add_text_field(col, stored=True, tokenizer_name="en_stem")
        schema_builder.add_text_field("_all", stored=False, tokenizer_name="en_stem")

        schema = schema_builder.build()
        self.tantivy_index = tantivy.Index(schema, path=str(self.tantivy_index_path))
        self.tantivy_index.reload()

        # Load FAISS indices
        self.faiss_indices = {}
        for embed_col in config.EMBEDDINGS_COLUMNS:
            index_file = self.faiss_index_path / f"{embed_col}.index"
            if index_file.exists():
                self.faiss_indices[embed_col] = faiss.read_index(str(index_file))

        # Load doc_id mappings
        self.doc_id_mappings = self.metadata['doc_id_mappings']

        # Load original dataframe for result retrieval
        self.df = pd.read_pickle(self.data_path)

    def _keyword_search_single(self, query: str, limit: int = 100) -> List[int]:
        """
        Perform keyword search for a single query string.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of document IDs
        """
        searcher = self.tantivy_index.searcher()

        # Search in all fields
        parsed_query = self.tantivy_index.parse_query(query, ["_all"])
        search_result = searcher.search(parsed_query, limit)

        doc_ids = []
        for (score, doc_address) in search_result.hits:
            doc = searcher.doc(doc_address)
            doc_id = doc.get_first("doc_id")
            if doc_id is not None:
                doc_ids.append(int(doc_id))

        return doc_ids

    def keyword_search(self, queries: Union[str, List[str]], limit: int = 100) -> Tuple[List[int], float]:
        """
        Perform keyword search with RRF ranking for multiple queries.

        Args:
            queries: Single query string or list of query strings
            limit: Maximum number of results

        Returns:
            Tuple of (ranked document IDs, search time in ms)
        """
        start_time = time.time()

        # Convert single query to list
        if isinstance(queries, str):
            queries = [queries]

        # Perform search for each query
        all_rankings = []
        for query in queries:
            if query and query.strip():
                doc_ids = self._keyword_search_single(query, limit)
                if doc_ids:
                    all_rankings.append(doc_ids)

        # Combine rankings using RRF
        if all_rankings:
            combined_ranking = reciprocal_rank_fusion(all_rankings)[:limit]
        else:
            combined_ranking = []

        search_time = (time.time() - start_time) * 1000  # Convert to ms
        return combined_ranking, search_time

    async def semantic_search(self, query: str, top_k: int = 5) -> Tuple[List[int], float]:
        """
        Perform semantic search using embeddings.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            Tuple of (document IDs, search time in ms)
        """
        start_time = time.time()

        # Get query embedding
        query_embedding = await self.embedding_client.get_embedding(query)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)

        all_results = []

        # Search in each embedding column
        for embed_col, index in self.faiss_indices.items():
            if index and embed_col in self.doc_id_mappings:
                # Perform search
                distances, indices = index.search(query_embedding, min(top_k, index.ntotal))

                # Map back to document IDs
                doc_id_mapping = self.doc_id_mappings[embed_col]
                for i, idx in enumerate(indices[0]):
                    if idx >= 0 and idx < len(doc_id_mapping):
                        doc_id = int(doc_id_mapping[idx])
                        score = 1.0 / (1.0 + distances[0][i])  # Convert distance to similarity score
                        all_results.append((doc_id, score))

        # Sort by score and get top-k
        all_results.sort(key=lambda x: x[1], reverse=True)
        doc_ids = [doc_id for doc_id, _ in all_results[:top_k]]

        search_time = (time.time() - start_time) * 1000  # Convert to ms
        return doc_ids, search_time

    async def combined_search(self, queries: List[str], top_k: int = 10) -> Tuple[List[int], float]:
        """
        Perform combined keyword and semantic search with RRF.

        Args:
            queries: List of search queries
            top_k: Number of results to return

        Returns:
            Tuple of (ranked document IDs, search time in ms)
        """
        start_time = time.time()

        # Perform keyword search
        keyword_results, _ = self.keyword_search(queries, limit=top_k * 2)

        # Perform semantic search for each query
        semantic_tasks = []
        for query in queries:
            if query and query.strip():
                semantic_tasks.append(self.semantic_search(query, top_k))

        semantic_results_list = await asyncio.gather(*semantic_tasks)

        # Combine all rankings
        all_rankings = []

        # Add keyword results
        if keyword_results:
            all_rankings.append(keyword_results)

        # Add semantic results
        for semantic_results, _ in semantic_results_list:
            if semantic_results:
                all_rankings.append(semantic_results)

        # Apply RRF to combine
        if all_rankings:
            combined_ranking = reciprocal_rank_fusion(all_rankings)[:top_k]
        else:
            combined_ranking = []

        search_time = (time.time() - start_time) * 1000  # Convert to ms
        return combined_ranking, search_time

    def get_documents(self, doc_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Retrieve full document data for given document IDs.

        Args:
            doc_ids: List of document IDs

        Returns:
            List of dictionaries containing document data
        """
        results = []
        for doc_id in doc_ids:
            if 0 <= doc_id < len(self.df):
                row = self.df.iloc[doc_id]
                # Convert row to dict and handle edge cases
                doc_dict = {}
                for col in self.df.columns:
                    value = row[col]
                    doc_dict[col] = handle_edge_cases(value)
                doc_dict['_doc_id'] = doc_id
                results.append(doc_dict)
        return results

    async def search(self,
                    query: Union[str, List[str]],
                    mode: str = "combined",
                    top_k: int = None) -> Dict[str, Any]:
        """
        Main search interface supporting all search modes.

        Args:
            query: Search query (string or list of strings)
            mode: Search mode - "keyword", "semantic", or "combined"
            top_k: Number of results to return

        Returns:
            Dictionary containing results and metadata
        """
        if top_k is None:
            top_k = config.DEFAULT_TOP_K

        # Validate mode
        if mode not in ["keyword", "semantic", "combined"]:
            raise ValueError(f"Invalid search mode: {mode}. Must be 'keyword', 'semantic', or 'combined'")

        # Perform search based on mode
        if mode == "keyword":
            doc_ids, search_time = self.keyword_search(query, limit=top_k)
        elif mode == "semantic":
            if isinstance(query, list):
                query = " ".join(query)  # Combine for semantic search
            doc_ids, search_time = await self.semantic_search(query, top_k=top_k)
        else:  # combined
            if isinstance(query, str):
                query = [query]
            doc_ids, search_time = await self.combined_search(query, top_k=top_k)

        # Check performance requirement
        if search_time > config.MAX_SEARCH_TIME_MS:
            print(f"Warning: Search took {search_time:.2f}ms, exceeding target of {config.MAX_SEARCH_TIME_MS}ms")

        # Get full documents
        documents = self.get_documents(doc_ids)

        return {
            "mode": mode,
            "query": query,
            "results": documents,
            "num_results": len(documents),
            "search_time_ms": round(search_time, 2),
            "performance_ok": search_time <= config.MAX_SEARCH_TIME_MS
        }


async def main():
    """Test the search engine."""
    engine = SearchEngine()

    # Test keyword search
    print("\n=== Keyword Search Test ===")
    result = await engine.search(["test", "example"], mode="keyword", top_k=5)
    print(f"Found {result['num_results']} results in {result['search_time_ms']}ms")

    # Test semantic search
    print("\n=== Semantic Search Test ===")
    result = await engine.search("find similar documents", mode="semantic", top_k=5)
    print(f"Found {result['num_results']} results in {result['search_time_ms']}ms")

    # Test combined search
    print("\n=== Combined Search Test ===")
    result = await engine.search(["test", "example"], mode="combined", top_k=5)
    print(f"Found {result['num_results']} results in {result['search_time_ms']}ms")


if __name__ == "__main__":
    asyncio.run(main())