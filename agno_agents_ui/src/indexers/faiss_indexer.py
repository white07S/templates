"""FAISS indexer for vector search."""

import os
import json
import pickle
import numpy as np
import faiss
import pyarrow.parquet as pq
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm


class FAISSIndexer:
    """Handle FAISS operations for vector-based search."""

    def __init__(self, index_path: str, vector_dimension: int = 4096):
        """Initialize FAISS indexer.

        Args:
            index_path: Path to store the FAISS index
            vector_dimension: Dimension of vectors
        """
        self.index_path = index_path
        self.vector_dimension = vector_dimension
        self.faiss_index_path = os.path.join(index_path, "faiss")
        self.indexes = {}  # column_name -> faiss.Index
        self.id_mappings = {}  # column_name -> list of doc_ids
        self.documents = []  # List of all documents

    def index_exists(self) -> bool:
        """Check if FAISS index exists."""
        meta_path = os.path.join(self.faiss_index_path, "meta.json")
        return os.path.exists(meta_path)

    def save_index(self, embedding_columns: List[str]):
        """Save FAISS indexes and metadata.

        Args:
            embedding_columns: List of columns with embeddings
        """
        os.makedirs(self.faiss_index_path, exist_ok=True)

        # Save each FAISS index
        for col in embedding_columns:
            if col in self.indexes:
                index_file = os.path.join(self.faiss_index_path, f"{col}.index")
                faiss.write_index(self.indexes[col], index_file)

                # Save ID mappings
                mapping_file = os.path.join(self.faiss_index_path, f"{col}_mapping.pkl")
                with open(mapping_file, 'wb') as f:
                    pickle.dump(self.id_mappings[col], f)

        # Save documents
        docs_file = os.path.join(self.faiss_index_path, "documents.pkl")
        with open(docs_file, 'wb') as f:
            pickle.dump(self.documents, f)

        # Save metadata
        meta = {
            "vector_dimension": self.vector_dimension,
            "embedding_columns": embedding_columns,
            "num_documents": len(self.documents)
        }
        meta_file = os.path.join(self.faiss_index_path, "meta.json")
        with open(meta_file, 'w') as f:
            json.dump(meta, f)

    def load_index(self):
        """Load existing FAISS indexes."""
        if not self.index_exists():
            raise ValueError(f"FAISS index not found at {self.faiss_index_path}")

        # Load metadata
        meta_file = os.path.join(self.faiss_index_path, "meta.json")
        with open(meta_file, 'r') as f:
            meta = json.load(f)

        self.vector_dimension = meta["vector_dimension"]
        embedding_columns = meta["embedding_columns"]

        # Load each FAISS index
        for col in embedding_columns:
            index_file = os.path.join(self.faiss_index_path, f"{col}.index")
            if os.path.exists(index_file):
                self.indexes[col] = faiss.read_index(index_file)

                # Load ID mappings
                mapping_file = os.path.join(self.faiss_index_path, f"{col}_mapping.pkl")
                with open(mapping_file, 'rb') as f:
                    self.id_mappings[col] = pickle.load(f)

        # Load documents
        docs_file = os.path.join(self.faiss_index_path, "documents.pkl")
        with open(docs_file, 'rb') as f:
            self.documents = pickle.load(f)

    def create_index_for_column(self, column_name: str, embeddings: np.ndarray, doc_ids: List[int]):
        """Create FAISS index for a specific column.

        Args:
            column_name: Name of the column
            embeddings: Numpy array of embeddings (N x D)
            doc_ids: List of document IDs corresponding to embeddings
        """
        print(f"Creating FAISS index for column '{column_name}'...")

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Create FAISS index
        # For production, use IVF index for speed with large datasets
        if len(embeddings) > 100000:
            # Use IVF index for large datasets
            nlist = min(4096, int(np.sqrt(len(embeddings))))  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.vector_dimension)  # Inner product for cosine similarity
            index = faiss.IndexIVFFlat(quantizer, self.vector_dimension, nlist, faiss.METRIC_INNER_PRODUCT)

            # Train the index
            print(f"Training IVF index with {nlist} clusters...")
            index.train(embeddings)
            index.add(embeddings)

            # Set search parameters
            index.nprobe = min(64, nlist // 4)  # Search this many clusters
        else:
            # Use flat index for smaller datasets
            index = faiss.IndexFlatIP(self.vector_dimension)  # Inner product for cosine similarity
            index.add(embeddings)

        self.indexes[column_name] = index
        self.id_mappings[column_name] = doc_ids

        print(f"✓ FAISS index created for '{column_name}' with {len(embeddings):,} vectors")

    def search_vector(self, query_vector: np.ndarray, column_name: str, k: int = 5) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Search for similar vectors in a specific column.

        Args:
            query_vector: Query vector (1D array)
            column_name: Column to search in
            k: Number of results to return

        Returns:
            List of (doc_id, score, document) tuples
        """
        if column_name not in self.indexes:
            return []

        # Reshape and normalize query vector
        query = np.array(query_vector).reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)

        # Search
        scores, indices = self.indexes[column_name].search(query, k)

        # Get results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0:  # Valid index
                doc_id = self.id_mappings[column_name][idx]
                document = self.documents[doc_id] if doc_id < len(self.documents) else {}
                results.append((doc_id, float(score), document))

        return results

    def search_multiple_columns(self, query_vector: np.ndarray, columns: List[str], k: int = 100) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Search across multiple columns.

        Args:
            query_vector: Query vector
            columns: List of columns to search
            k: Number of results per column

        Returns:
            List of (doc_id, score, document) tuples
        """
        all_results = []

        for col in columns:
            if col in self.indexes:
                results = self.search_vector(query_vector, col, k)
                all_results.extend(results)

        return all_results

    def create_dummy_embeddings(self, num_docs: int, embedding_columns: List[str], documents: List[Dict]) -> Dict[str, np.ndarray]:
        """Create dummy random embeddings for testing.

        Args:
            num_docs: Number of documents
            embedding_columns: Columns to create embeddings for
            documents: List of document dictionaries

        Returns:
            Dictionary of column_name -> embeddings array
        """
        print("Creating dummy embeddings for testing...")
        embeddings_dict = {}

        for col in tqdm(embedding_columns, desc="Creating embeddings"):
            # Create random embeddings
            embeddings = np.random.randn(num_docs, self.vector_dimension).astype('float32')
            embeddings_dict[col] = embeddings

        return embeddings_dict

    def build_from_parquet(self, parquet_path: str, embedding_columns: List[str], force_reindex: bool = False):
        """Build FAISS index from parquet file with dummy embeddings.

        Args:
            parquet_path: Path to parquet file
            embedding_columns: Columns to create embeddings for
            force_reindex: Whether to force reindexing
        """
        if self.index_exists() and not force_reindex:
            print("FAISS index already exists. Loading...")
            self.load_index()
            return

        print("Building FAISS indexes...")

        # Read parquet file
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        # Store documents
        print("Storing documents...")
        self.documents = []
        for idx in tqdm(range(len(df)), desc="Processing documents"):
            row_dict = df.iloc[idx].to_dict()
            # Convert non-serializable types
            for key, val in row_dict.items():
                if hasattr(val, 'isoformat'):
                    row_dict[key] = val.isoformat()
                elif isinstance(val, (float, int)) and str(val) in ['nan', 'NaN']:
                    row_dict[key] = None
            self.documents.append(row_dict)

        # Create dummy embeddings for each column
        embeddings_dict = self.create_dummy_embeddings(len(df), embedding_columns, self.documents)

        # Create FAISS index for each embedding column
        for col in embedding_columns:
            if col in embeddings_dict:
                doc_ids = list(range(len(df)))
                self.create_index_for_column(col, embeddings_dict[col], doc_ids)

        # Save indexes
        self.save_index(embedding_columns)

        print(f"✓ FAISS indexes created for {len(embedding_columns)} columns")