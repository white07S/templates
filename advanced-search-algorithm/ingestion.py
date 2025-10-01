"""
Ingestion and indexing module for the advanced search engine.
Handles data loading, embedding generation, and index creation.
"""

import os
import json
import pickle
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
import duckdb
import tantivy
import faiss
from tqdm import tqdm
import httpx
from dotenv import load_dotenv
import hashlib
import pyarrow.parquet as pq

# Load environment variables
load_dotenv()

class DataIngestionEngine:
    def __init__(
        self,
        data_path: str,
        index_path: str,
        embedding_columns: List[str] = None,
        vector_dimension: int = 4096,
        use_real_embeddings: bool = False,
        openai_key: str = None,
        openai_url: str = None,
        openai_model: str = None,
        re_index: bool = False,
        chunk_size: int = 1000
    ):
        """
        Initialize the data ingestion engine.

        Args:
            data_path: Path to the parquet file
            index_path: Path to store index files
            embedding_columns: Columns to convert to embeddings
            vector_dimension: Dimension of embedding vectors
            use_real_embeddings: Whether to use real embeddings or random
            openai_key: OpenAI API key (for real embeddings)
            openai_url: OpenAI API URL (for self-hosted)
            openai_model: Model name for embeddings
            re_index: Force re-indexing
            chunk_size: Batch size for processing
        """
        self.data_path = data_path
        self.index_path = Path(index_path)
        self.embedding_columns = embedding_columns or []
        self.vector_dimension = vector_dimension
        self.use_real_embeddings = use_real_embeddings
        self.openai_key = openai_key
        self.openai_url = openai_url or "https://api.openai.com/v1"
        self.openai_model = openai_model or "text-embedding-3-large"
        self.re_index = re_index
        self.chunk_size = chunk_size

        # Create index directory if not exists
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Paths for different index components
        self.duckdb_path = self.index_path / "duckdb_index.db"
        self.tantivy_path = self.index_path / "tantivy_index"
        self.faiss_path = self.index_path / "faiss_indices"
        self.metadata_path = self.index_path / "metadata.json"
        self.embedding_cache_path = self.index_path / "embedding_cache.pkl"

        # Initialize components
        self.conn = None
        self.tantivy_index = None
        self.faiss_indices = {}
        self.metadata = {}
        self.embedding_cache = {}

    def _load_embedding_cache(self):
        """Load embedding cache if exists."""
        if self.embedding_cache_path.exists():
            try:
                with open(self.embedding_cache_path, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                print(f"Loaded embedding cache with {len(self.embedding_cache)} entries")
            except Exception as e:
                print(f"Failed to load embedding cache: {e}")
                self.embedding_cache = {}

    def _save_embedding_cache(self):
        """Save embedding cache to disk."""
        with open(self.embedding_cache_path, 'wb') as f:
            pickle.dump(self.embedding_cache, f)

    def _get_text_hash(self, text: str) -> str:
        """Get hash of text for caching."""
        return hashlib.md5(str(text).encode()).hexdigest()

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        if not self.use_real_embeddings:
            # Generate random embedding for testing
            np.random.seed(hash(text) % (2**32))
            return np.random.randn(self.vector_dimension).astype(np.float32)

        # Check cache first
        text_hash = self._get_text_hash(text)
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]

        # Call OpenAI API for real embeddings
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json"
        }

        data = {
            "input": text,
            "model": self.openai_model,
            "encoding_format": "float"
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.openai_url}/embeddings",
                    headers=headers,
                    json=data
                )
                response.raise_for_status()
                embedding = np.array(response.json()["data"][0]["embedding"], dtype=np.float32)

                # Cache the embedding
                self.embedding_cache[text_hash] = embedding
                return embedding

        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Fallback to random embedding
            return np.random.randn(self.vector_dimension).astype(np.float32)

    def _check_existing_indices(self) -> bool:
        """Check if indices already exist."""
        if self.re_index:
            return False

        required_paths = [
            self.duckdb_path,
            self.tantivy_path,
            self.metadata_path
        ]

        # Check FAISS indices for embedding columns
        for col in self.embedding_columns:
            faiss_index_file = self.faiss_path / f"{col}.index"
            required_paths.append(faiss_index_file)

        return all(path.exists() for path in required_paths)

    def ingest_and_index(self) -> Dict[str, Any]:
        """
        Main ingestion and indexing pipeline.

        Returns:
            Dictionary with ingestion statistics
        """
        # Check if indices exist
        if self._check_existing_indices():
            print("Indices already exist. Skipping ingestion.")
            print("Set re_index=True to force re-indexing.")

            # Load metadata
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)

            return self.metadata

        print("Starting data ingestion and indexing...")

        # Clean up existing indices if re-indexing
        if self.re_index and self.index_path.exists():
            print("Cleaning up existing indices...")
            shutil.rmtree(self.index_path)
            self.index_path.mkdir(parents=True, exist_ok=True)

        # Load embedding cache
        self._load_embedding_cache()

        # Load data
        print(f"Loading data from {self.data_path}...")
        df = pd.read_parquet(self.data_path)
        total_rows = len(df)
        total_columns = len(df.columns)

        print(f"Data loaded: {total_rows:,} rows, {total_columns} columns")

        # Store metadata
        self.metadata = {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "embedding_columns": self.embedding_columns,
            "vector_dimension": self.vector_dimension
        }

        # 1. Create DuckDB index
        self._create_duckdb_index(df)

        # 2. Create Tantivy full-text index
        self._create_tantivy_index(df)

        # 3. Create FAISS vector indices
        if self.embedding_columns:
            self._create_faiss_indices(df)

        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        # Save embedding cache
        self._save_embedding_cache()

        print("\nIngestion and indexing completed successfully!")
        return self.metadata

    def _create_duckdb_index(self, df: pd.DataFrame):
        """Create DuckDB index for structured queries."""
        print("\nCreating DuckDB index...")

        # Connect to DuckDB
        self.conn = duckdb.connect(str(self.duckdb_path))

        # Create table from DataFrame with progress bar
        with tqdm(total=1, desc="Loading data into DuckDB") as pbar:
            self.conn.execute("CREATE TABLE documents AS SELECT * FROM df")
            pbar.update(1)

        # Create indices on commonly queried columns
        index_columns = [
            'id', 'status', 'type', 'priority', 'region', 'product',
            'category', 'department', 'is_active', 'created_date'
        ]

        for col in tqdm(index_columns, desc="Creating DuckDB indices"):
            if col in df.columns:
                try:
                    self.conn.execute(f"CREATE INDEX idx_{col} ON documents({col})")
                except Exception as e:
                    print(f"Warning: Could not create index on {col}: {e}")

        # Analyze table for query optimization
        self.conn.execute("ANALYZE documents")

        print(f"DuckDB index created at {self.duckdb_path}")

    def _create_tantivy_index(self, df: pd.DataFrame):
        """Create Tantivy full-text search index."""
        print("\nCreating Tantivy full-text index...")

        # Create schema
        schema_builder = tantivy.SchemaBuilder()

        # Add document ID
        schema_builder.add_integer_field("doc_id", stored=True)

        # Add text fields for all string columns
        text_columns = df.select_dtypes(include=['object']).columns.tolist()

        for col in text_columns:
            schema_builder.add_text_field(col, stored=False, tokenizer_name="default")

        schema = schema_builder.build()

        # Create index
        if self.tantivy_path.exists():
            shutil.rmtree(self.tantivy_path)

        self.tantivy_path.mkdir(parents=True, exist_ok=True)
        index = tantivy.Index(schema, path=str(self.tantivy_path))

        # Add documents with progress bar
        writer = index.writer(heap_size=100_000_000)  # 100MB heap

        batch_size = 10000
        num_batches = (len(df) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Indexing documents in Tantivy"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))

            for idx in range(start_idx, end_idx):
                row = df.iloc[idx]
                doc = tantivy.Document()
                doc.add_integer("doc_id", idx)

                # Add text fields
                for col in text_columns:
                    value = row[col]
                    if pd.notna(value) and value != '':
                        doc.add_text(col, str(value))

                writer.add_document(doc)

            # Commit periodically
            if batch_idx % 10 == 0:
                writer.commit()

        writer.commit()
        print(f"Tantivy index created at {self.tantivy_path}")

    def _create_faiss_indices(self, df: pd.DataFrame):
        """Create FAISS indices for vector search."""
        print(f"\nCreating FAISS indices for {len(self.embedding_columns)} embedding columns...")

        self.faiss_path.mkdir(parents=True, exist_ok=True)

        for col in self.embedding_columns:
            print(f"\nProcessing embeddings for column: {col}")

            if col not in df.columns:
                print(f"Warning: Column {col} not found in data")
                continue

            # Generate embeddings for the column
            embeddings = []

            # Process in batches to show progress
            batch_size = 1000
            num_batches = (len(df) + batch_size - 1) // batch_size

            for batch_idx in tqdm(range(num_batches), desc=f"Generating embeddings for {col}"):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(df))

                batch_embeddings = []
                for idx in range(start_idx, end_idx):
                    text = str(df.iloc[idx][col])
                    if pd.notna(text) and text != '' and text != 'nan':
                        embedding = self._generate_embedding(text)
                    else:
                        # Use zero vector for null values
                        embedding = np.zeros(self.vector_dimension, dtype=np.float32)

                    batch_embeddings.append(embedding)

                embeddings.extend(batch_embeddings)

                # Save cache periodically
                if batch_idx % 10 == 0:
                    self._save_embedding_cache()

            # Convert to numpy array
            embeddings = np.array(embeddings, dtype=np.float32)

            # Create FAISS index
            print(f"Building FAISS index for {col}...")
            index = faiss.IndexFlatL2(self.vector_dimension)

            # Add embeddings to index
            index.add(embeddings)

            # Save index
            index_file = self.faiss_path / f"{col}.index"
            faiss.write_index(index, str(index_file))

            # Save embeddings for later use
            embeddings_file = self.faiss_path / f"{col}_embeddings.npy"
            np.save(embeddings_file, embeddings)

            print(f"FAISS index for {col} created with {index.ntotal} vectors")

        print("All FAISS indices created successfully")

def main():
    """Main function for testing."""
    # Load configuration from environment
    load_dotenv()

    # Configuration
    config = {
        "data_path": os.getenv("DATA_PATH", "mock_data_small.parquet"),
        "index_path": os.getenv("INDEX_PATH", "./indices"),
        "embedding_columns": os.getenv("EMBEDDING_COLUMNS", "title,description,content,article").split(","),
        "vector_dimension": int(os.getenv("VECTOR_DIMENSION", "4096")),
        "use_real_embeddings": os.getenv("USE_REAL_EMBEDDINGS", "false").lower() == "true",
        "openai_key": os.getenv("OPENAI_API_KEY"),
        "openai_url": os.getenv("OPENAI_API_URL"),
        "openai_model": os.getenv("OPENAI_MODEL"),
        "re_index": os.getenv("RE_INDEX", "false").lower() == "true",
        "chunk_size": int(os.getenv("CHUNK_SIZE", "1000"))
    }

    # Create ingestion engine
    engine = DataIngestionEngine(**config)

    # Run ingestion and indexing
    metadata = engine.ingest_and_index()

    print("\nIngestion Summary:")
    print(f"Total rows indexed: {metadata['total_rows']:,}")
    print(f"Total columns: {metadata['total_columns']}")
    print(f"Embedding columns: {metadata['embedding_columns']}")
    print(f"Vector dimension: {metadata['vector_dimension']}")

if __name__ == "__main__":
    main()