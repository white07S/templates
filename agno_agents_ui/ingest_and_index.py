"""Main ingestion and indexing pipeline for the search engine."""

import os
import time
import pickle
import numpy as np
from typing import List, Optional
from tqdm import tqdm
import argparse

from src.core.config import IndexConfig
from src.indexers.duckdb_indexer import DuckDBIndexer
from src.indexers.tantivy_indexer import TantivyIndexer
from src.indexers.faiss_indexer import FAISSIndexer
from src.utils.generate_mock_data import generate_mock_dataset


class IngestionPipeline:
    """Main pipeline for data ingestion and indexing."""

    def __init__(self, config: IndexConfig):
        """Initialize the ingestion pipeline.

        Args:
            config: Index configuration
        """
        self.config = config

        # Initialize indexers
        self.duckdb_indexer = DuckDBIndexer(config.index_path)
        self.tantivy_indexer = TantivyIndexer(config.index_path)
        self.faiss_indexer = FAISSIndexer(config.index_path, config.vector_dimension)

        # Embedding cache
        self.embedding_cache = {}
        self.cache_path = config.embedding_cache_path

    def load_embedding_cache(self):
        """Load embedding cache from disk."""
        if os.path.exists(self.cache_path):
            print("Loading embedding cache...")
            with open(self.cache_path, 'rb') as f:
                self.embedding_cache = pickle.load(f)
            print(f"Loaded {len(self.embedding_cache)} cached embeddings")

    def save_embedding_cache(self):
        """Save embedding cache to disk."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.embedding_cache, f)
        print(f"Saved {len(self.embedding_cache)} embeddings to cache")

    def generate_embeddings(self, texts: List[str], column_name: str) -> np.ndarray:
        """Generate embeddings for texts (dummy implementation for now).

        Args:
            texts: List of texts to embed
            column_name: Name of the column being embedded

        Returns:
            Numpy array of embeddings
        """
        embeddings = []

        for text in tqdm(texts, desc=f"Generating embeddings for {column_name}"):
            # Check cache
            cache_key = f"{column_name}:{text[:100]}"  # Use first 100 chars as key

            if cache_key in self.embedding_cache:
                embeddings.append(self.embedding_cache[cache_key])
            else:
                # Generate dummy embedding for now
                # In production, this would call OpenAI API
                embedding = np.random.randn(self.config.vector_dimension).astype('float32')
                embeddings.append(embedding)
                self.embedding_cache[cache_key] = embedding

        return np.array(embeddings, dtype='float32')

    def check_indexes_exist(self) -> bool:
        """Check if all indexes already exist."""
        return (
            self.duckdb_indexer.index_exists() and
            self.tantivy_indexer.index_exists() and
            self.faiss_indexer.index_exists()
        )

    def run(self):
        """Run the full ingestion and indexing pipeline."""
        print("=" * 80)
        print("SEARCH ENGINE INGESTION AND INDEXING PIPELINE")
        print("=" * 80)

        # Check if indexes exist
        if self.check_indexes_exist() and not self.config.re_index:
            print("\n✓ All indexes already exist. Use --reindex flag to force reindexing.")
            return

        # Ensure data exists
        if not os.path.exists(self.config.data_path):
            print(f"\n⚠ Data file not found at {self.config.data_path}")
            print("Generating mock dataset...")
            generate_mock_dataset(num_rows=10_000_000, output_path=self.config.data_path)

        # Load embedding cache if reindexing
        if self.config.re_index:
            self.load_embedding_cache()

        # Step 1: Create DuckDB index
        print("\n" + "=" * 60)
        print("STEP 1: Creating DuckDB Index")
        print("=" * 60)
        start_time = time.time()
        self.duckdb_indexer.create_index(self.config.data_path, self.config.re_index)
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        # Step 2: Create Tantivy index
        print("\n" + "=" * 60)
        print("STEP 2: Creating Tantivy Index")
        print("=" * 60)
        start_time = time.time()
        self.tantivy_indexer.create_index(self.config.data_path, self.config.re_index)
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        # Step 3: Create FAISS indexes
        print("\n" + "=" * 60)
        print("STEP 3: Creating FAISS Indexes")
        print("=" * 60)
        start_time = time.time()

        # Use default embedding columns if not specified
        if not self.config.embedding_columns:
            self.config.embedding_columns = ["content", "abstract", "summary", "body"]
            print(f"Using default embedding columns: {self.config.embedding_columns}")

        self.faiss_indexer.build_from_parquet(
            self.config.data_path,
            self.config.embedding_columns,
            self.config.re_index
        )
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        # Save embedding cache
        if self.config.re_index:
            self.save_embedding_cache()

        print("\n" + "=" * 80)
        print("✓ INGESTION AND INDEXING COMPLETE")
        print("=" * 80)

        # Print summary
        print("\nIndex Summary:")
        print(f"  - Index path: {self.config.index_path}")
        print(f"  - DuckDB rows: {self.duckdb_indexer.get_row_count():,}")
        print(f"  - Tantivy index: {self.tantivy_indexer.index_path}")
        print(f"  - FAISS indexes: {len(self.config.embedding_columns)} columns")
        print(f"  - Vector dimension: {self.config.vector_dimension}")


def main():
    """Main entry point for the ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Ingest and index data for search engine")

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/data.parquet",
        help="Path to the parquet data file"
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="data/index",
        help="Path to store index files"
    )
    parser.add_argument(
        "--embedding-columns",
        type=str,
        nargs="+",
        default=["content", "abstract", "summary", "body"],
        help="Columns to create embeddings for"
    )
    parser.add_argument(
        "--vector-dimension",
        type=int,
        default=4096,
        help="Dimension of embedding vectors"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force reindexing even if indexes exist"
    )
    parser.add_argument(
        "--openai-key",
        type=str,
        default=None,
        help="OpenAI API key (optional, uses dummy embeddings if not provided)"
    )

    args = parser.parse_args()

    # Create configuration
    config = IndexConfig(
        data_path=args.data_path,
        index_path=args.index_path,
        embedding_columns=args.embedding_columns,
        vector_dimension=args.vector_dimension,
        re_index=args.reindex,
        openai_api_key=args.openai_key
    )

    # Run pipeline
    pipeline = IngestionPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()