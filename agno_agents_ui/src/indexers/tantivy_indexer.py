"""Tantivy indexer for keyword search."""

import os
import json
import pyarrow.parquet as pq
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import tantivy
import shutil


class TantivyIndexer:
    """Handle Tantivy operations for keyword-based search."""

    def __init__(self, index_path: str):
        """Initialize Tantivy indexer.

        Args:
            index_path: Path to store the Tantivy index
        """
        self.index_path = os.path.join(index_path, "tantivy_index")
        self.index = None
        self.searcher = None
        self.schema = None

    def index_exists(self) -> bool:
        """Check if Tantivy index exists."""
        meta_path = os.path.join(self.index_path, "meta.json")
        return os.path.exists(meta_path)

    def _build_schema(self, columns: List[str]) -> tantivy.Schema:
        """Build Tantivy schema based on columns.

        Args:
            columns: List of column names

        Returns:
            Tantivy schema
        """
        schema_builder = tantivy.SchemaBuilder()

        # Add document ID field (stored and indexed)
        schema_builder.add_integer_field("doc_id", indexed=True, stored=True)

        # Add text fields for all columns (searchable)
        for col in columns:
            # All columns are added as text fields for keyword search
            schema_builder.add_text_field(col, stored=False, tokenizer_name="en_stem")

        # Add a special field to store the full document as JSON
        schema_builder.add_json_field("_source", stored=True)

        return schema_builder.build()

    def create_index(self, parquet_path: str, force_reindex: bool = False):
        """Create Tantivy index from parquet file.

        Args:
            parquet_path: Path to the parquet file
            force_reindex: Whether to force reindexing
        """
        if self.index_exists() and not force_reindex:
            print("Tantivy index already exists. Skipping...")
            return

        print("Creating Tantivy index...")

        # Remove old index if force reindex
        if force_reindex and os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)

        os.makedirs(self.index_path, exist_ok=True)

        # Read parquet file
        print("Reading parquet file...")
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        # Get columns (exclude those that might not be text-searchable)
        text_columns = []
        for col in df.columns:
            # Include all string-like columns
            if df[col].dtype == 'object' or str(df[col].dtype).startswith('string'):
                text_columns.append(col)

        print(f"Indexing {len(text_columns)} text columns...")

        # Build schema
        self.schema = self._build_schema(text_columns)

        # Create index
        self.index = tantivy.Index(schema=self.schema, path=str(self.index_path))

        # Get writer
        writer = self.index.writer(heap_size=500_000_000, num_threads=4)

        # Index documents in batches
        batch_size = 10000
        num_batches = (len(df) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Indexing documents"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))

            for idx in range(start_idx, end_idx):
                row = df.iloc[idx]

                # Create Tantivy document
                doc = tantivy.Document()

                # Add doc_id
                doc.add_integer("doc_id", int(idx))

                # Add text fields
                for col in text_columns:
                    value = row[col]
                    if value is not None and str(value) not in ['nan', 'NaN', '', 'None']:
                        # Convert to string and add to document
                        doc.add_text(col, str(value))

                # Store full document as JSON
                row_dict = row.to_dict()
                # Convert any non-serializable types
                for key, val in row_dict.items():
                    if hasattr(val, 'isoformat'):
                        row_dict[key] = val.isoformat()
                    elif isinstance(val, (float, int)) and str(val) in ['nan', 'NaN']:
                        row_dict[key] = None

                doc.add_json("_source", json.dumps(row_dict))

                # Add document to index
                writer.add_document(doc)

        # Commit changes
        print("Committing index...")
        writer.commit()
        writer.wait_merging_threads()

        print(f"✓ Tantivy index created with {len(df):,} documents")

    def load_index(self):
        """Load existing Tantivy index."""
        if not self.index_exists():
            raise ValueError(f"Tantivy index not found at {self.index_path}")

        # Load schema from meta.json
        meta_path = os.path.join(self.index_path, "meta.json")
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # Rebuild schema from meta
        schema_builder = tantivy.SchemaBuilder()
        for field in meta['schema']:
            if field['name'] == 'doc_id':
                schema_builder.add_integer_field("doc_id", indexed=True, stored=True)
            elif field['name'] == '_source':
                schema_builder.add_json_field("_source", stored=True)
            else:
                schema_builder.add_text_field(field['name'], stored=False, tokenizer_name="en_stem")

        self.schema = schema_builder.build()

        # Open index
        self.index = tantivy.Index(schema=self.schema, path=str(self.index_path))
        self.index.reload()
        self.searcher = self.index.searcher()

    def search_keywords(self, keywords: List[str], limit: int = 5) -> List[tuple[float, Dict[str, Any]]]:
        """Search documents using keywords.

        Args:
            keywords: List of keywords to search
            limit: Maximum number of results

        Returns:
            List of (score, document) tuples
        """
        if not self.searcher:
            self.load_index()

        # Get searchable fields (exclude doc_id and _source)
        searchable_fields = []
        meta_path = os.path.join(self.index_path, "meta.json")
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            for field in meta['schema']:
                if field['name'] not in ['doc_id', '_source']:
                    searchable_fields.append(field['name'])

        # Build query string
        query_string = " ".join(keywords)

        # Parse query
        query = self.index.parse_query(query_string, searchable_fields)

        # Search
        search_result = self.searcher.search(query, limit)

        # Extract results
        results = []
        for score, doc_address in search_result.hits:
            doc = self.searcher.doc(doc_address)
            # Parse the stored JSON document
            source_json = doc.get("_source")[0] if doc.get("_source") else "{}"
            source_doc = json.loads(source_json)
            results.append((score, source_doc))

        return results

    def search_in_columns(self, keywords: List[str], columns: List[str], limit: int = 100) -> List[tuple[int, float, Dict[str, Any]]]:
        """Search keywords in specific columns.

        Args:
            keywords: List of keywords to search
            columns: List of columns to search in
            limit: Maximum results per column

        Returns:
            List of (doc_id, score, document) tuples
        """
        if not self.searcher:
            self.load_index()

        results = []
        query_string = " ".join(keywords)

        # Search in each column
        for col in columns:
            try:
                query = self.index.parse_query(query_string, [col])
                search_result = self.searcher.search(query, limit)

                for score, doc_address in search_result.hits:
                    doc = self.searcher.doc(doc_address)
                    doc_id = doc.get("doc_id")[0] if doc.get("doc_id") else -1
                    source_json = doc.get("_source")[0] if doc.get("_source") else "{}"
                    source_doc = json.loads(source_json)
                    results.append((doc_id, score, source_doc))
            except Exception as e:
                # Column might not exist in schema
                continue

        return results