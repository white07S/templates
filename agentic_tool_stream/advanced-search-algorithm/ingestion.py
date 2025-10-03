"""
Data ingestion and indexing module for the advanced search algorithm.
"""
import asyncio
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import faiss
import tantivy
from tqdm import tqdm
import config
from utils import EmbeddingClient, handle_edge_cases, prepare_text_for_indexing


class DataIngestion:
    """Handle data ingestion and index building."""

    def __init__(self, force_reindex: bool = False):
        """
        Initialize the data ingestion system.

        Args:
            force_reindex: Force rebuilding all indexes even if they exist
        """
        self.force_reindex = force_reindex or config.REINDEX
        self.embedding_client = EmbeddingClient()
        self.tantivy_index_path = Path(config.INDEX_PATH) / "tantivy"
        self.faiss_index_path = Path(config.INDEX_PATH) / "faiss"
        self.metadata_path = Path(config.INDEX_PATH) / "metadata.pkl"

    def _clean_index_directory(self):
        """Clean existing index directories if reindexing."""
        if self.force_reindex:
            for path in [self.tantivy_index_path, self.faiss_index_path]:
                if path.exists():
                    shutil.rmtree(path)
            if self.metadata_path.exists():
                os.remove(self.metadata_path)

    def _load_data(self) -> pd.DataFrame:
        """Load data from parquet file."""
        try:
            df = pd.read_parquet(config.DATA_PATH)
            print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def _create_tantivy_schema(self, columns: List[str]) -> tantivy.Schema:
        """Create Tantivy schema based on dataframe columns."""
        schema_builder = tantivy.SchemaBuilder()

        # Add document ID field
        schema_builder.add_integer_field("doc_id", indexed=True, stored=True)

        # Add all columns as text fields
        for col in columns:
            schema_builder.add_text_field(col, stored=True, tokenizer_name="en_stem")

        # Add combined searchable field
        schema_builder.add_text_field("_all", stored=False, tokenizer_name="en_stem")

        return schema_builder.build()

    def _build_tantivy_index(self, df: pd.DataFrame) -> tantivy.Index:
        """Build Tantivy index for keyword search."""
        print("Building Tantivy index...")

        # Create schema
        schema = self._create_tantivy_schema(df.columns.tolist())

        # Create index directory
        self.tantivy_index_path.mkdir(parents=True, exist_ok=True)

        # Create index
        index = tantivy.Index(schema, path=str(self.tantivy_index_path))

        # Create writer
        writer = index.writer(heap_size=config.TANTIVY_HEAP_SIZE, num_threads=config.PARALLEL_WORKERS)

        # Add documents in chunks
        total_rows = len(df)
        chunk_size = config.CHUNK_SIZE

        for chunk_start in tqdm(range(0, total_rows, chunk_size), desc="Indexing documents"):
            chunk_end = min(chunk_start + chunk_size, total_rows)
            chunk_df = df.iloc[chunk_start:chunk_end]

            for idx, row in chunk_df.iterrows():
                doc = tantivy.Document()

                # Add document ID
                doc.add_integer("doc_id", int(idx))

                # Add all fields
                combined_text = []
                for col in df.columns:
                    value = handle_edge_cases(row[col])
                    if value:
                        doc.add_text(col, value)
                        combined_text.append(value)

                # Add combined field for full-text search
                if combined_text:
                    doc.add_text("_all", " ".join(combined_text))

                writer.add_document(doc)

        # Commit and wait for merge
        writer.commit()
        writer.wait_merging_threads()

        print("Tantivy index built successfully")
        return index

    async def _build_faiss_index(self, df: pd.DataFrame) -> tuple:
        """
        Build FAISS index for semantic search.

        Returns:
            Tuple of (faiss_indices, doc_id_mappings)
        """
        print("Building FAISS indices...")

        faiss_indices = {}
        doc_id_mappings = {}

        for embed_col in config.EMBEDDINGS_COLUMNS:
            print(f"Building FAISS index for column: {embed_col}")

            # Prepare texts for embedding
            texts = []
            valid_indices = []

            for idx, row in df.iterrows():
                text = handle_edge_cases(row.get(embed_col, ""))
                if text:
                    texts.append(text)
                    valid_indices.append(idx)

            if not texts:
                print(f"No valid texts found for column {embed_col}, skipping...")
                continue

            # Get embeddings
            embeddings = await self.embedding_client.get_embeddings(texts)

            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)

            # Normalize for cosine similarity (L2 distance on normalized vectors = cosine distance)
            faiss.normalize_L2(embeddings_array)

            # Create FAISS index
            index = faiss.IndexFlatL2(config.VECTOR_DIMENSIONS)
            index.add(embeddings_array)

            # Store index and mapping
            faiss_indices[embed_col] = index
            doc_id_mappings[embed_col] = np.array(valid_indices, dtype=np.int64)

            print(f"Added {len(valid_indices)} vectors to FAISS index for {embed_col}")

        return faiss_indices, doc_id_mappings

    def _save_indices(self, tantivy_index: tantivy.Index, faiss_indices: Dict, doc_id_mappings: Dict, df: pd.DataFrame):
        """Save all indices and metadata to disk."""
        print("Saving indices...")

        # Tantivy index is already saved at creation

        # Save FAISS indices
        self.faiss_index_path.mkdir(parents=True, exist_ok=True)
        for col_name, index in faiss_indices.items():
            index_file = self.faiss_index_path / f"{col_name}.index"
            faiss.write_index(index, str(index_file))

        # Save metadata
        import pickle
        metadata = {
            'doc_id_mappings': doc_id_mappings,
            'columns': df.columns.tolist(),
            'total_docs': len(df)
        }
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        # Save the dataframe for result retrieval
        df_path = Path(config.INDEX_PATH) / "data.pkl"
        df.to_pickle(df_path)

        print("All indices saved successfully")

    async def ingest_and_index(self) -> bool:
        """
        Main ingestion and indexing process.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Clean if reindexing
            self._clean_index_directory()

            # Check if indices already exist
            if not self.force_reindex and self.tantivy_index_path.exists() and self.faiss_index_path.exists():
                print("Indices already exist. Use force_reindex=True to rebuild.")
                return True

            # Load data
            df = self._load_data()

            # Build Tantivy index for keyword search
            tantivy_index = self._build_tantivy_index(df)

            # Build FAISS indices for semantic search
            faiss_indices, doc_id_mappings = await self._build_faiss_index(df)

            # Save all indices
            self._save_indices(tantivy_index, faiss_indices, doc_id_mappings, df)

            print("Ingestion and indexing completed successfully")
            return True

        except Exception as e:
            print(f"Error during ingestion: {e}")
            return False


async def main():
    """Main function for testing ingestion."""
    ingestion = DataIngestion(force_reindex=True)
    success = await ingestion.ingest_and_index()
    if success:
        print("Ingestion completed successfully!")
    else:
        print("Ingestion failed!")


if __name__ == "__main__":
    asyncio.run(main())