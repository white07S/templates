"""
Data Ingestion and Indexing Pipeline
Handles parquet file ingestion, embeddings generation, and index creation
"""

import os
import json
import pickle
import duckdb
import faiss
import tantivy
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from embeddings_client import EmbeddingsClient
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import hashlib

class IngestionPipeline:
    def __init__(
        self,
        data_path: str,
        index_path: str = "indices",
        embedding_columns: Optional[List[str]] = None,
        vector_dimension: int = 4096,
        openai_key: Optional[str] = None,
        openai_url: Optional[str] = None,
        openai_model: str = "text-embedding-3-large",
        batch_size: int = 10000,
        max_workers: int = 4,
        re_index: bool = False
    ):
        self.data_path = data_path
        self.index_path = index_path
        self.embedding_columns = embedding_columns or []
        self.vector_dimension = vector_dimension
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.re_index = re_index

        # Create index directory
        os.makedirs(self.index_path, exist_ok=True)

        # Initialize embeddings client
        self.embeddings_client = EmbeddingsClient(
            api_key=openai_key,
            api_url=openai_url,
            model=openai_model,
            dimension=vector_dimension,
            cache_dir=os.path.join(index_path, "embeddings_cache"),
            use_cache=True,
            max_workers=max_workers
        )

        # Index paths
        self.duckdb_path = os.path.join(index_path, "duckdb.db")
        self.tantivy_path = os.path.join(index_path, "tantivy")
        self.faiss_indices_path = os.path.join(index_path, "faiss")
        self.metadata_path = os.path.join(index_path, "metadata.json")

        # Check if indices already exist
        self.indices_exist = self._check_indices_exist()

    def _check_indices_exist(self) -> bool:
        """Check if all required indices exist"""
        return all([
            os.path.exists(self.duckdb_path),
            os.path.exists(self.tantivy_path),
            os.path.exists(self.faiss_indices_path),
            os.path.exists(self.metadata_path)
        ])

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from disk"""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save metadata to disk"""
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _get_data_hash(self) -> str:
        """Get hash of data file for change detection"""
        with open(self.data_path, 'rb') as f:
            # Read first and last 1MB for hash
            file_hash = hashlib.sha256()
            file_hash.update(f.read(1024 * 1024))
            f.seek(-1024 * 1024, 2)
            file_hash.update(f.read(1024 * 1024))
            return file_hash.hexdigest()

    def _build_duckdb_index(self, df: pd.DataFrame, append: bool = False):
        """Build DuckDB index for dataframe operations"""
        conn = duckdb.connect(self.duckdb_path)

        # Create or replace table
        table_name = "search_data"
        if not append:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")

        # Register DataFrame as a view
        conn.register('df_view', df)

        # Create table from DataFrame
        if append:
            conn.execute(f"INSERT INTO {table_name} SELECT * FROM df_view")
        else:
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df_view")

        # Create indices on common search columns
        # Get column names and types
        columns = conn.execute(f"DESCRIBE {table_name}").fetchall()

        for col_name, col_type, *_ in columns:
            if 'VARCHAR' in col_type or 'TEXT' in col_type:
                # Create index for text columns
                try:
                    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{col_name} ON {table_name}({col_name})")
                except:
                    pass  # Some columns might not be suitable for indexing

        # Optimize the database
        conn.execute("CHECKPOINT")
        conn.close()

    def _build_tantivy_index(self, df: pd.DataFrame):
        """Build Tantivy full-text search index"""
        # Create Tantivy schema
        schema_builder = tantivy.SchemaBuilder()

        # Add document ID
        schema_builder.add_integer_field("doc_id", stored=True, indexed=True)

        # Add text fields for searchable columns
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':  # Text columns
                try:
                    # Check if column contains text data
                    sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else ""
                    if isinstance(sample, str):
                        schema_builder.add_text_field(col, stored=False, tokenizer_name="default")
                        text_columns.append(col)
                except:
                    pass

        # Build schema and create index
        schema = schema_builder.build()

        # Remove existing index if re-indexing
        if os.path.exists(self.tantivy_path):
            shutil.rmtree(self.tantivy_path)
        os.makedirs(self.tantivy_path, exist_ok=True)

        # Create index
        index = tantivy.Index(schema, path=self.tantivy_path)
        writer = index.writer(heap_size=500_000_000)  # 500MB heap

        # Add documents
        print("Building Tantivy index...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Indexing documents"):
            doc = tantivy.Document()
            doc.add_integer("doc_id", int(idx))

            # Add text fields
            for col in text_columns:
                value = row[col]
                if pd.notna(value) and value != "":
                    doc.add_text(col, str(value))

            writer.add_document(doc)

            # Commit periodically
            if idx % 10000 == 0:
                writer.commit()

        writer.commit()
        index.reload()

        # Save text columns list
        with open(os.path.join(self.tantivy_path, "text_columns.json"), 'w') as f:
            json.dump(text_columns, f)

    def _build_faiss_indices(self, df: pd.DataFrame):
        """Build FAISS indices for vector search"""
        os.makedirs(self.faiss_indices_path, exist_ok=True)

        if not self.embedding_columns:
            print("No embedding columns specified. Skipping FAISS index creation.")
            return

        faiss_metadata = {}

        for col in self.embedding_columns:
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found in data. Skipping.")
                continue

            print(f"Building FAISS index for column: {col}")

            # Get embeddings for this column
            texts = df[col].fillna("").tolist()
            embeddings = self.embeddings_client.get_embeddings_batch(
                texts,
                show_progress=True,
                desc=f"Generating embeddings for {col}"
            )

            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)

            # Create FAISS index
            # Using IVF index for scalability
            nlist = min(100, len(embeddings_array) // 100)  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.vector_dimension)
            index = faiss.IndexIVFFlat(quantizer, self.vector_dimension, nlist)

            # Train index
            print(f"Training FAISS index for {col}...")
            index.train(embeddings_array)

            # Add vectors
            index.add(embeddings_array)

            # Save index
            index_path = os.path.join(self.faiss_indices_path, f"{col}.index")
            faiss.write_index(index, index_path)

            faiss_metadata[col] = {
                "num_vectors": len(embeddings_array),
                "dimension": self.vector_dimension,
                "index_type": "IVFFlat",
                "nlist": nlist
            }

        # Save FAISS metadata
        with open(os.path.join(self.faiss_indices_path, "metadata.json"), 'w') as f:
            json.dump(faiss_metadata, f, indent=2)

    def ingest(self):
        """Main ingestion pipeline"""
        # Check if indices exist and re_index is False
        if self.indices_exist and not self.re_index:
            print("Indices already exist. Use re_index=True to rebuild.")
            return

        print(f"Starting ingestion pipeline...")
        print(f"Data path: {self.data_path}")
        print(f"Index path: {self.index_path}")

        # Load parquet file
        print("Loading parquet file...")
        parquet_file = pq.ParquetFile(self.data_path)
        total_rows = parquet_file.metadata.num_rows
        print(f"Total rows: {total_rows:,}")

        # Process in batches for memory efficiency
        num_batches = (total_rows + self.batch_size - 1) // self.batch_size

        # Initialize indices
        first_batch = True

        for batch_num in tqdm(range(num_batches), desc="Processing batches"):
            # Read batch
            batch_start = batch_num * self.batch_size
            batch_end = min((batch_num + 1) * self.batch_size, total_rows)

            # Read batch from parquet
            batch_df = parquet_file.read_row_group(batch_num).to_pandas()

            # Build DuckDB index (append after first batch)
            with tqdm.external_write_mode():
                print(f"\nBuilding DuckDB index for batch {batch_num + 1}/{num_batches}...")
            self._build_duckdb_index(batch_df, append=(not first_batch))

            # Build Tantivy index (only on first batch for now)
            if first_batch:
                with tqdm.external_write_mode():
                    print(f"\nBuilding Tantivy index...")
                self._build_tantivy_index(batch_df)

            # Build FAISS indices (only on first batch for now)
            if first_batch and self.embedding_columns:
                with tqdm.external_write_mode():
                    print(f"\nBuilding FAISS indices...")
                self._build_faiss_indices(batch_df)

            first_batch = False

            # Break after first batch for testing
            if batch_num == 0 and total_rows > self.batch_size:
                print(f"\nProcessed first batch of {self.batch_size} rows for testing.")
                print("Full ingestion would process all {total_rows:,} rows.")
                break

        # Save metadata
        metadata = {
            "data_path": self.data_path,
            "data_hash": self._get_data_hash(),
            "total_rows": total_rows,
            "embedding_columns": self.embedding_columns,
            "vector_dimension": self.vector_dimension,
            "index_created": pd.Timestamp.now().isoformat()
        }
        self._save_metadata(metadata)

        # Save embeddings cache
        if self.embeddings_client.use_cache:
            self.embeddings_client._save_cache()

        print("\nIngestion pipeline completed successfully!")
        print(f"Indices saved to: {self.index_path}")

        # Print cache statistics
        cache_stats = self.embeddings_client.get_cache_stats()
        print(f"\nEmbeddings cache statistics:")
        print(f"  - Cached embeddings: {cache_stats['cache_size']}")
        print(f"  - Cache file: {cache_stats['cache_file']}")

if __name__ == "__main__":
    # Test ingestion with small dataset
    pipeline = IngestionPipeline(
        data_path="data/mock_data_1m.parquet",
        index_path="indices_test",
        embedding_columns=["long_text_1", "medium_text_1"],  # Test with 2 columns
        vector_dimension=4096,
        re_index=True,
        batch_size=50000
    )
    pipeline.ingest()