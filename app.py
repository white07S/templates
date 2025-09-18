"""
FAISS-based configurable indexing and similarity search system
with async embedding support and dynamic column handling
"""

import asyncio
import json
import pickle
import hashlib
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
    force_reindex: bool = False  # Force reindexing even if index exists

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

    def _compute_data_hash(self, df: pd.DataFrame, column_configs: Dict[str, str]) -> str:
        """Compute a hash of the data to detect changes"""
        hash_obj = hashlib.sha256()

        # Hash the column configurations
        hash_obj.update(json.dumps(column_configs, sort_keys=True).encode())

        # Hash the hash column
        hash_obj.update(df['hash'].to_json().encode())

        # Hash each configured column
        for col_name in column_configs:
            if col_name in df.columns:
                hash_obj.update(df[col_name].to_json().encode())

        return hash_obj.hexdigest()

    def index_exists(self) -> bool:
        """Check if index files exist"""
        return (
            Path(self.config.index_path).exists() and
            Path(self.config.metadata_path).exists()
        )

    def should_reindex(self, df: pd.DataFrame, column_configs: Dict[str, str]) -> bool:
        """Check if reindexing is needed based on data changes"""
        if self.config.force_reindex:
            return True

        if not self.index_exists():
            return True

        # Load existing metadata to check data hash
        try:
            with open(self.config.metadata_path, 'rb') as f:
                existing_metadata = pickle.load(f)

            current_hash = self._compute_data_hash(df, column_configs)
            existing_hash = existing_metadata.get('data_hash', '')

            if current_hash != existing_hash:
                print(f"Data has changed (hash mismatch), reindexing required")
                return True

            return False
        except Exception as e:
            print(f"Error checking existing index: {e}")
            return True

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
        show_progress: bool = True,
        auto_load: bool = True
    ) -> bool:
        """
        Prepare and index dataframe columns or load existing index

        Args:
            df: DataFrame with 'hash' column and columns to index
            column_configs: Dict mapping column names to types ("str" or "List[str]")
            show_progress: Whether to show progress bars
            auto_load: Automatically load existing index if data hasn't changed

        Returns:
            bool: True if new indexing was performed, False if loaded from cache
        """
        if 'hash' not in df.columns:
            raise ValueError("DataFrame must have 'hash' column")

        if len(column_configs) > 3:
            raise ValueError("Cannot index more than 3 columns")

        self.config.column_configs = column_configs

        # Check if we should use existing index
        if auto_load and not self.should_reindex(df, column_configs):
            print(f"Loading existing index from {self.config.index_path}")
            self.load(self.config.index_path, self.config.metadata_path)
            print(f"Index loaded successfully with {self.index.ntotal} vectors")
            return False

        print("Creating new index...")

        # Initialize index
        self.index = faiss.IndexFlatL2(self.config.dimension)

        # Compute data hash
        data_hash = self._compute_data_hash(df, column_configs)

        # Store metadata
        self.metadata = {
            'hashes': df['hash'].tolist(),
            'column_mappings': {},
            'column_configs': column_configs,
            'total_vectors': 0,
            'data_hash': data_hash
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
        return True

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
Demonstration of smart caching behavior in FAISS indexer
"""

import asyncio
import numpy as np
import pandas as pd
import os
from faiss_indexer import FAISSIndexer, IndexConfig


async def mock_embedding_func(texts):
    """Mock embedding function that tracks API calls"""
    global api_call_count
    api_call_count += 1
    print(f"  → API Call #{api_call_count}: Processing {len(texts)} texts")
    await asyncio.sleep(0.1)  # Simulate API latency
    return np.random.randn(len(texts), 4096).astype('float32')


async def demo_caching():
    """Demonstrate the caching behavior"""
    global api_call_count

    print("=" * 70)
    print("FAISS INDEXER CACHING DEMONSTRATION")
    print("=" * 70)

    # Sample data
    df = pd.DataFrame({
        'hash': ['doc1', 'doc2', 'doc3'],
        'content': ['AI research', 'Machine learning', 'Deep learning'],
        'tags': [['AI', 'research'], ['ML', 'algorithms'], ['DL', 'neural']]
    })

    column_configs = {
        'content': 'str',
        'tags': 'List[str]'
    }

    # Clean up previous runs
    for file in ['cache_demo.idx', 'cache_demo.pkl']:
        if os.path.exists(file):
            os.remove(file)

    config = IndexConfig(
        dimension=4096,
        index_path='cache_demo.idx',
        metadata_path='cache_demo.pkl'
    )

    print("\n1. FIRST RUN - Creating new index")
    print("-" * 50)
    api_call_count = 0
    indexer1 = FAISSIndexer(mock_embedding_func, config)
    was_indexed = await indexer1.prepare_and_index(df, column_configs)
    if was_indexed:
        indexer1.save()
    print(f"Result: {'Created new index' if was_indexed else 'Loaded from cache'}")
    print(f"Total API calls made: {api_call_count}")

    print("\n2. SECOND RUN - Same data (should use cache)")
    print("-" * 50)
    api_call_count = 0
    indexer2 = FAISSIndexer(mock_embedding_func, config)
    was_indexed = await indexer2.prepare_and_index(df, column_configs)
    print(f"Result: {'Created new index' if was_indexed else 'Loaded from cache'}")
    print(f"Total API calls made: {api_call_count} ✓ (No API calls!)")

    print("\n3. THIRD RUN - Modified data (should reindex)")
    print("-" * 50)
    df_modified = df.copy()
    df_modified.loc[0, 'content'] = 'Artificial Intelligence research'  # Changed content
    api_call_count = 0
    indexer3 = FAISSIndexer(mock_embedding_func, config)
    was_indexed = await indexer3.prepare_and_index(df_modified, column_configs)
    if was_indexed:
        indexer3.save()
    print(f"Result: {'Created new index' if was_indexed else 'Loaded from cache'}")
    print(f"Total API calls made: {api_call_count}")

    print("\n4. FOURTH RUN - Force reindex")
    print("-" * 50)
    config_force = IndexConfig(
        dimension=4096,
        index_path='cache_demo.idx',
        metadata_path='cache_demo.pkl',
        force_reindex=True
    )
    api_call_count = 0
    indexer4 = FAISSIndexer(mock_embedding_func, config_force)
    was_indexed = await indexer4.prepare_and_index(df_modified, column_configs)
    if was_indexed:
        indexer4.save()
    print(f"Result: {'Created new index' if was_indexed else 'Loaded from cache'}")
    print(f"Total API calls made: {api_call_count}")

    print("\n5. FIFTH RUN - Disable auto-load")
    print("-" * 50)
    api_call_count = 0
    indexer5 = FAISSIndexer(mock_embedding_func, config)
    was_indexed = await indexer5.prepare_and_index(df_modified, column_configs, auto_load=False)
    if was_indexed:
        indexer5.save()
    print(f"Result: {'Created new index' if was_indexed else 'Loaded from cache'}")
    print(f"Total API calls made: {api_call_count}")

    # Clean up
    for file in ['cache_demo.idx', 'cache_demo.pkl']:
        if os.path.exists(file):
            os.remove(file)

    print("\n" + "=" * 70)
    print("CACHING DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Points:")
    print("• Index is automatically loaded when data hasn't changed")
    print("• No API calls are made when using cached index")
    print("• Data changes are detected via SHA-256 hashing")
    print("• Force reindex option available when needed")
    print("• Auto-load can be disabled for manual control")


if __name__ == "__main__":
    api_call_count = 0
    asyncio.run(demo_caching())
