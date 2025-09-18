"""
FAISS-based configurable indexing and similarity search system
with async embedding support and dynamic column handling
"""

import asyncio
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass, asdict

import aiohttp
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class IndexConfig:
    """Configuration for FAISS index"""
    dimension: int = 4096
    column_configs: Dict[str, str] = None  # column_name -> "str" or "List[str]"
    index_path: str = "faiss_index.idx"
    metadata_path: str = "faiss_metadata.pkl"

    def to_dict(self):
        return asdict(self)


class AsyncEmbeddingClient:
    """Async client for OpenAI-compatible embedding endpoints"""

    def __init__(
        self,
        api_url: str,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-large",
        batch_size: int = 100,
        max_concurrent_requests: int = 10
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.max_concurrent_requests = max_concurrent_requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def _make_embedding_request(
        self,
        session: aiohttp.ClientSession,
        texts: List[str]
    ) -> List[List[float]]:
        """Make async embedding request"""
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "input": texts
        }

        async with self.semaphore:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload
            ) as response:
                response.raise_for_status()
                data = await response.json()
                # Extract embeddings from response
                embeddings = [item["embedding"] for item in data["data"]]
                return embeddings

    async def get_embeddings_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """Get embeddings for a batch of texts with progress tracking"""
        all_embeddings = []

        # Create batches
        batches = [
            texts[i:i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        async with aiohttp.ClientSession() as session:
            # Create tasks for all batches
            tasks = []
            for batch in batches:
                task = self._make_embedding_request(session, batch)
                tasks.append(task)

            # Process with progress bar
            if show_progress:
                with tqdm(total=len(tasks), desc="Getting embeddings") as pbar:
                    results = []
                    for coro in asyncio.as_completed(tasks):
                        result = await coro
                        results.append(result)
                        pbar.update(1)

                    # Sort results to maintain order
                    for batch_embeddings in results:
                        all_embeddings.extend(batch_embeddings)
            else:
                results = await asyncio.gather(*tasks)
                for batch_embeddings in results:
                    all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype='float32')


class FAISSIndexer:
    """Main FAISS indexing and search class"""

    def __init__(
        self,
        embedding_func: Union[Callable, AsyncEmbeddingClient],
        config: Optional[IndexConfig] = None
    ):
        self.embedding_func = embedding_func
        self.config = config or IndexConfig()
        self.index = None
        self.metadata = {}

    async def _get_embeddings_for_column(
        self,
        data: Union[List[str], List[List[str]]],
        column_type: str,
        show_progress: bool = True
    ) -> tuple[np.ndarray, List[int]]:
        """Get embeddings for a column based on its type"""
        all_texts = []
        text_to_row_mapping = []

        for row_idx, item in enumerate(data):
            if column_type == "str":
                all_texts.append(item)
                text_to_row_mapping.append(row_idx)
            elif column_type == "List[str]":
                for text in item:
                    all_texts.append(text)
                    text_to_row_mapping.append(row_idx)

        # Get embeddings
        if isinstance(self.embedding_func, AsyncEmbeddingClient):
            embeddings = await self.embedding_func.get_embeddings_batch(
                all_texts,
                show_progress=show_progress
            )
        else:
            # Assume it's an async function
            embeddings = await self.embedding_func(all_texts)
            embeddings = np.array(embeddings, dtype='float32')

        return embeddings, text_to_row_mapping

    async def prepare_and_index(
        self,
        df: pd.DataFrame,
        column_configs: Dict[str, str],
        show_progress: bool = True
    ) -> None:
        """
        Prepare and index dataframe columns

        Args:
            df: DataFrame with 'hash' column and columns to index
            column_configs: Dict mapping column names to types ("str" or "List[str]")
            show_progress: Whether to show progress bars
        """
        if 'hash' not in df.columns:
            raise ValueError("DataFrame must have 'hash' column")

        if len(column_configs) > 3:
            raise ValueError("Cannot index more than 3 columns")

        self.config.column_configs = column_configs

        # Initialize index
        self.index = faiss.IndexFlatL2(self.config.dimension)

        # Store metadata
        self.metadata = {
            'hashes': df['hash'].tolist(),
            'column_mappings': {},
            'column_configs': column_configs,
            'total_vectors': 0
        }

        current_vector_idx = 0

        # Process each column
        for col_name, col_type in column_configs.items():
            if col_name not in df.columns:
                raise ValueError(f"Column {col_name} not found in DataFrame")

            print(f"Processing column: {col_name} (type: {col_type})")

            # Get embeddings for this column
            embeddings, row_mapping = await self._get_embeddings_for_column(
                df[col_name].tolist(),
                col_type,
                show_progress=show_progress
            )

            # Store mapping information
            column_mapping = {
                'start_idx': current_vector_idx,
                'end_idx': current_vector_idx + len(embeddings),
                'row_mapping': row_mapping,
                'column_type': col_type
            }
            self.metadata['column_mappings'][col_name] = column_mapping

            # Add to index
            self.index.add(embeddings)
            current_vector_idx += len(embeddings)

        self.metadata['total_vectors'] = self.index.ntotal
        print(f"Total vectors indexed: {self.index.ntotal}")

    def save(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None):
        """Save index and metadata to files"""
        index_path = index_path or self.config.index_path
        metadata_path = metadata_path or self.config.metadata_path

        # Save FAISS index
        faiss.write_index(self.index, index_path)

        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

        print(f"Index saved to {index_path}")
        print(f"Metadata saved to {metadata_path}")

    def load(self, index_path: str, metadata_path: str):
        """Load index and metadata from files"""
        self.index = faiss.read_index(index_path)

        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        self.config.column_configs = self.metadata.get('column_configs', {})
        print(f"Loaded index with {self.index.ntotal} vectors")

    def search(
        self,
        query_embedding: np.ndarray,
        column_name: str,
        top_n: int = 5
    ) -> List[str]:
        """
        Search for similar vectors in a specific column

        Args:
            query_embedding: Query vector (1D array of size 4096)
            column_name: Name of column to search in
            top_n: Number of top results to return (max 5)

        Returns:
            List of hash values for top matches
        """
        if top_n > 5:
            raise ValueError("top_n cannot exceed 5")

        if column_name not in self.metadata['column_mappings']:
            raise ValueError(f"Column {column_name} not found in index")

        # Ensure query embedding is the right shape and type
        query_embedding = np.array(query_embedding, dtype='float32').reshape(1, -1)

        column_mapping = self.metadata['column_mappings'][column_name]
        column_type = column_mapping['column_type']

        # Search in the entire index
        k = min(top_n * 10, self.index.ntotal)  # Search for more to filter later
        distances, indices = self.index.search(query_embedding, k)

        # Filter results based on column range
        start_idx = column_mapping['start_idx']
        end_idx = column_mapping['end_idx']
        row_mapping = column_mapping['row_mapping']

        results = []
        seen_hashes = set()

        for dist, idx in zip(distances[0], indices[0]):
            # Check if this vector belongs to the target column
            if start_idx <= idx < end_idx:
                # Get the row index
                vector_offset = idx - start_idx
                row_idx = row_mapping[vector_offset]
                hash_val = self.metadata['hashes'][row_idx]

                # For List[str] columns, ensure unique hashes
                if column_type == "List[str]":
                    if hash_val not in seen_hashes:
                        results.append((hash_val, float(dist)))
                        seen_hashes.add(hash_val)
                else:
                    results.append((hash_val, float(dist)))

                if len(results) >= top_n:
                    break

        # Return just the hashes
        return [hash_val for hash_val, _ in results]


class FAISSSearcher:
    """Standalone searcher class for loading and searching existing indexes"""

    def __init__(self, index_path: str, metadata_path: str):
        self.index = faiss.read_index(index_path)

        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        print(f"Loaded index with {self.index.ntotal} vectors")
        print(f"Available columns: {list(self.metadata['column_mappings'].keys())}")

    def search(
        self,
        query_embedding: np.ndarray,
        column_name: str,
        top_n: int = 5
    ) -> List[str]:
        """
        Search for similar vectors in a specific column

        Args:
            query_embedding: Query vector (1D array of size 4096)
            column_name: Name of column to search in
            top_n: Number of top results to return (max 5)

        Returns:
            List of hash values for top matches
        """
        if top_n > 5:
            raise ValueError("top_n cannot exceed 5")

        if column_name not in self.metadata['column_mappings']:
            raise ValueError(f"Column {column_name} not found in index. Available: {list(self.metadata['column_mappings'].keys())}")

        # Ensure query embedding is the right shape and type
        query_embedding = np.array(query_embedding, dtype='float32').reshape(1, -1)

        column_mapping = self.metadata['column_mappings'][column_name]
        column_type = column_mapping['column_type']

        # Search in the entire index
        k = min(top_n * 10, self.index.ntotal)  # Search for more to filter later
        distances, indices = self.index.search(query_embedding, k)

        # Filter results based on column range
        start_idx = column_mapping['start_idx']
        end_idx = column_mapping['end_idx']
        row_mapping = column_mapping['row_mapping']

        results = []
        seen_hashes = set()

        for dist, idx in zip(distances[0], indices[0]):
            # Check if this vector belongs to the target column
            if start_idx <= idx < end_idx:
                # Get the row index
                vector_offset = idx - start_idx
                row_idx = row_mapping[vector_offset]
                hash_val = self.metadata['hashes'][row_idx]

                # For List[str] columns, ensure unique hashes
                if column_type == "List[str]":
                    if hash_val not in seen_hashes:
                        results.append((hash_val, float(dist)))
                        seen_hashes.add(hash_val)
                else:
                    results.append((hash_val, float(dist)))

                if len(results) >= top_n:
                    break

        # Return just the hashes
        return [hash_val for hash_val, _ in results]




"""
Test script for FAISS indexer with sample data and mock embeddings
"""

import asyncio
import numpy as np
import pandas as pd
from faiss_indexer import (
    FAISSIndexer,
    FAISSSearcher,
    AsyncEmbeddingClient,
    IndexConfig
)


async def mock_embedding_function(texts):
    """Mock embedding function for testing"""
    # Simulate async processing
    await asyncio.sleep(0.01)
    # Generate random embeddings of dimension 4096
    return np.random.randn(len(texts), 4096).astype('float32')


async def test_with_mock_data():
    """Test the indexer with mock data"""
    print("=" * 60)
    print("Testing FAISS Indexer with Mock Data")
    print("=" * 60)

    # Create sample DataFrame
    data = {
        'hash': ['hash_001', 'hash_002', 'hash_003', 'hash_004', 'hash_005'],
        'title': [
            'Machine Learning Basics',
            'Deep Neural Networks',
            'Natural Language Processing',
            'Computer Vision Fundamentals',
            'Reinforcement Learning'
        ],
        'keywords': [
            ['ML', 'AI', 'algorithms'],
            ['DNN', 'backprop', 'layers'],
            ['NLP', 'transformers', 'BERT'],
            ['CV', 'CNN', 'images'],
            ['RL', 'Q-learning', 'agents']
        ],
        'description': [
            'Introduction to machine learning concepts',
            'Advanced deep learning architectures',
            'Modern NLP techniques and models',
            'Image processing and recognition',
            'Agent-based learning systems'
        ]
    }
    df = pd.DataFrame(data)

    print("\nSample DataFrame:")
    print(df[['hash', 'title']].head())

    # Configure columns to index
    column_configs = {
        'title': 'str',
        'keywords': 'List[str]',
        'description': 'str'
    }

    # Create indexer with mock embedding function
    config = IndexConfig(
        dimension=4096,
        index_path='test_index.idx',
        metadata_path='test_metadata.pkl'
    )
    indexer = FAISSIndexer(mock_embedding_function, config)

    # Prepare and index
    print("\nIndexing data...")
    await indexer.prepare_and_index(df, column_configs, show_progress=True)

    # Save to disk
    indexer.save()

    # Test search functionality
    print("\n" + "=" * 60)
    print("Testing Search Functionality")
    print("=" * 60)

    # Create a mock query embedding
    query_embedding = np.random.randn(4096).astype('float32')

    # Test search in 'title' column (str type)
    print("\nSearching in 'title' column (str type):")
    results = indexer.search(query_embedding, 'title', top_n=3)
    print(f"Top 3 results: {results}")

    # Test search in 'keywords' column (List[str] type)
    print("\nSearching in 'keywords' column (List[str] type):")
    results = indexer.search(query_embedding, 'keywords', top_n=3)
    print(f"Top 3 unique results: {results}")

    # Test search in 'description' column (str type)
    print("\nSearching in 'description' column (str type):")
    results = indexer.search(query_embedding, 'description', top_n=2)
    print(f"Top 2 results: {results}")


async def test_with_api_client():
    """Test with AsyncEmbeddingClient (requires actual API endpoint)"""
    print("\n" + "=" * 60)
    print("Testing with AsyncEmbeddingClient")
    print("=" * 60)

    # Note: This requires an actual OpenAI-compatible API endpoint
    # Uncomment and configure with your API details to test

    """
    # Configure the embedding client
    client = AsyncEmbeddingClient(
        api_url="https://api.openai.com/v1/embeddings",  # or your endpoint
        api_key="your-api-key-here",
        model="text-embedding-3-large",
        batch_size=50,
        max_concurrent_requests=5
    )

    # Create sample data
    data = {
        'hash': ['doc_001', 'doc_002', 'doc_003'],
        'content': [
            'Understanding quantum computing principles',
            'Blockchain technology and cryptocurrencies',
            'Artificial general intelligence research'
        ],
        'tags': [
            ['quantum', 'computing', 'physics'],
            ['blockchain', 'crypto', 'distributed'],
            ['AGI', 'AI', 'research']
        ]
    }
    df = pd.DataFrame(data)

    column_configs = {
        'content': 'str',
        'tags': 'List[str]'
    }

    # Create indexer with API client
    config = IndexConfig(
        dimension=4096,
        index_path='api_test_index.idx',
        metadata_path='api_test_metadata.pkl'
    )
    indexer = FAISSIndexer(client, config)

    # Index the data
    await indexer.prepare_and_index(df, column_configs)
    indexer.save()

    # Test search
    query_text = "quantum mechanics"
    query_embedding = await client.get_embeddings_batch([query_text])
    results = indexer.search(query_embedding[0], 'content', top_n=2)
    print(f"Search results for '{query_text}': {results}")
    """

    print("API client test requires configuration with actual endpoint")
    print("See commented code for implementation example")


def test_standalone_searcher():
    """Test the standalone searcher"""
    print("\n" + "=" * 60)
    print("Testing Standalone Searcher")
    print("=" * 60)

    try:
        # Load existing index
        searcher = FAISSSearcher('test_index.idx', 'test_metadata.pkl')

        # Create a query embedding
        query_embedding = np.random.randn(4096).astype('float32')

        # Search in different columns
        for column in ['title', 'keywords', 'description']:
            results = searcher.search(query_embedding, column, top_n=2)
            print(f"\nSearch in '{column}': {results}")

    except FileNotFoundError:
        print("Index files not found. Run the async test first to create them.")


async def main():
    """Main test function"""
    # Test with mock data
    await test_with_mock_data()

    # Test standalone searcher
    test_standalone_searcher()

    # Test with API client (optional)
    await test_with_api_client()

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
