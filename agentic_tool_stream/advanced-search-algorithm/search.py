"""
Main search interface for the advanced search algorithm.
"""
import asyncio
import json
from typing import Union, List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import config
from ingestion import DataIngestion
from engine import SearchEngine


class SearchInterface:
    """High-level interface for the search system."""

    def __init__(self, auto_ingest: bool = True):
        """
        Initialize the search interface.

        Args:
            auto_ingest: Automatically run ingestion if indices don't exist
        """
        self.auto_ingest = auto_ingest
        self.engine = None
        self._initialize()

    def _initialize(self):
        """Initialize the search system."""
        # Check if indices exist
        index_path = Path(config.INDEX_PATH)
        tantivy_path = index_path / "tantivy"
        faiss_path = index_path / "faiss"

        if not tantivy_path.exists() or not faiss_path.exists():
            if self.auto_ingest:
                print("Indices not found. Running ingestion...")
                asyncio.run(self._run_ingestion())
            else:
                raise ValueError("Indices not found. Please run ingestion first.")

        # Initialize search engine
        try:
            self.engine = SearchEngine()
            print("Search engine initialized successfully")
        except Exception as e:
            print(f"Error initializing search engine: {e}")
            raise

    async def _run_ingestion(self):
        """Run data ingestion and indexing."""
        ingestion = DataIngestion(force_reindex=False)
        success = await ingestion.ingest_and_index()
        if not success:
            raise RuntimeError("Ingestion failed")

    async def keyword_search(self,
                            queries: Union[str, List[str]],
                            top_k: int = None) -> Dict[str, Any]:
        """
        Perform keyword-based search.

        Args:
            queries: Search query or list of queries
            top_k: Number of results to return

        Returns:
            Search results dictionary
        """
        if not self.engine:
            raise RuntimeError("Search engine not initialized")

        return await self.engine.search(queries, mode="keyword", top_k=top_k)

    async def semantic_search(self,
                             query: str,
                             top_k: int = None) -> Dict[str, Any]:
        """
        Perform semantic similarity search.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            Search results dictionary
        """
        if not self.engine:
            raise RuntimeError("Search engine not initialized")

        return await self.engine.search(query, mode="semantic", top_k=top_k)

    async def combined_search(self,
                             queries: Union[str, List[str]],
                             top_k: int = None) -> Dict[str, Any]:
        """
        Perform combined keyword and semantic search.

        Args:
            queries: Search query or list of queries
            top_k: Number of results to return

        Returns:
            Search results dictionary
        """
        if not self.engine:
            raise RuntimeError("Search engine not initialized")

        return await self.engine.search(queries, mode="combined", top_k=top_k)

    def print_results(self, results: Dict[str, Any], max_display: int = 3):
        """
        Pretty print search results.

        Args:
            results: Search results dictionary
            max_display: Maximum number of results to display
        """
        print(f"\n{'='*60}")
        print(f"Search Mode: {results['mode'].upper()}")
        print(f"Query: {results['query']}")
        print(f"Found: {results['num_results']} results")
        print(f"Search Time: {results['search_time_ms']}ms", end="")
        if results['performance_ok']:
            print(" ✓ (within 150ms target)")
        else:
            print(" ✗ (exceeded 150ms target)")
        print(f"{'='*60}")

        # Display results
        for i, doc in enumerate(results['results'][:max_display]):
            print(f"\n--- Result {i+1} (Doc ID: {doc.get('_doc_id', 'N/A')}) ---")

            # Show first few non-empty fields
            fields_shown = 0
            for key, value in doc.items():
                if key.startswith('_'):
                    continue
                if value and str(value).strip():
                    print(f"{key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
                    fields_shown += 1
                    if fields_shown >= 3:
                        break

        if results['num_results'] > max_display:
            print(f"\n... and {results['num_results'] - max_display} more results")

    def export_results(self, results: Dict[str, Any], output_path: str = "search_results.json"):
        """
        Export search results to JSON file.

        Args:
            results: Search results dictionary
            output_path: Path to save the results
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results exported to {output_path}")


async def main():
    """Main function for testing the search interface."""
    # Initialize search interface
    print("Initializing search system...")
    search = SearchInterface(auto_ingest=True)

    # Example searches
    test_queries = {
        "keyword": ["data", "analysis", "machine learning"],
        "semantic": "find documents about artificial intelligence and neural networks",
        "combined": ["python", "programming", "algorithms"]
    }

    # Test keyword search
    print("\n" + "="*60)
    print("KEYWORD SEARCH TEST")
    print("="*60)
    results = await search.keyword_search(test_queries["keyword"], top_k=5)
    search.print_results(results)

    # Test semantic search
    print("\n" + "="*60)
    print("SEMANTIC SEARCH TEST")
    print("="*60)
    results = await search.semantic_search(test_queries["semantic"], top_k=5)
    search.print_results(results)

    # Test combined search
    print("\n" + "="*60)
    print("COMBINED SEARCH TEST")
    print("="*60)
    results = await search.combined_search(test_queries["combined"], top_k=5)
    search.print_results(results)

    # Export last results
    search.export_results(results, "sample_results.json")


async def benchmark():
    """Benchmark the search performance."""
    print("Running performance benchmark...")
    search = SearchInterface(auto_ingest=False)

    # Prepare test queries
    keyword_queries = [
        ["test"],
        ["test", "data"],
        ["test", "data", "analysis"],
        ["machine", "learning", "algorithm", "model"],
        ["python", "code", "function", "class", "method"]
    ]

    semantic_queries = [
        "artificial intelligence",
        "data science and machine learning",
        "software development best practices",
        "cloud computing infrastructure",
        "natural language processing"
    ]

    # Benchmark keyword search
    print("\n=== Keyword Search Benchmark ===")
    total_time = 0
    for queries in keyword_queries:
        result = await search.keyword_search(queries, top_k=10)
        print(f"Query: {queries[:2]}{'...' if len(queries) > 2 else ''} - "
              f"Time: {result['search_time_ms']}ms - "
              f"Results: {result['num_results']}")
        total_time += result['search_time_ms']
    avg_time = total_time / len(keyword_queries)
    print(f"Average keyword search time: {avg_time:.2f}ms")

    # Benchmark semantic search
    print("\n=== Semantic Search Benchmark ===")
    total_time = 0
    for query in semantic_queries:
        result = await search.semantic_search(query, top_k=5)
        print(f"Query: {query[:30]}... - "
              f"Time: {result['search_time_ms']}ms - "
              f"Results: {result['num_results']}")
        total_time += result['search_time_ms']
    avg_time = total_time / len(semantic_queries)
    print(f"Average semantic search time: {avg_time:.2f}ms")

    # Benchmark combined search
    print("\n=== Combined Search Benchmark ===")
    total_time = 0
    for i, queries in enumerate(keyword_queries):
        result = await search.combined_search(queries, top_k=10)
        print(f"Query {i+1} - "
              f"Time: {result['search_time_ms']}ms - "
              f"Results: {result['num_results']}")
        total_time += result['search_time_ms']
    avg_time = total_time / len(keyword_queries)
    print(f"Average combined search time: {avg_time:.2f}ms")

    print(f"\nTarget performance: <{config.MAX_SEARCH_TIME_MS}ms per search")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        asyncio.run(benchmark())
    else:
        asyncio.run(main())