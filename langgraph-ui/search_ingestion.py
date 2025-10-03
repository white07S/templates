"""
Data ingestion module for creating Tantivy and FAISS indices.
"""
import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# Tantivy imports
try:
    import tantivy
except ImportError:
    print("Please install tantivy-py: uv add tantivy")
    raise

# FAISS imports
try:
    import faiss
except ImportError:
    print("Please install faiss-cpu: uv add faiss-cpu")
    raise

from config import SearchConfig
from utils import (
    OptimizedEmbeddingClient,
    DataProcessor,
    PerformanceMonitor,
    clean_text,
    handle_edge_cases,
    logger
)


class TantivyIndexer:
    """Handle Tantivy index creation and management."""
    
    def __init__(self, index_path: str = SearchConfig.TANTIVY_INDEX_PATH):
        self.index_path = Path(index_path)
        self.index = None
        self.schema = None
        self.writer = None
    
    def create_schema(self, columns: List[str]) -> tantivy.Schema:
        """Create Tantivy schema based on dataframe columns."""
        schema_builder = tantivy.SchemaBuilder()
        
        # Add document ID field (always stored)
        schema_builder.add_integer_field("doc_id", stored=True, indexed=True)
        
        # Add text fields for each column
        for col in columns:
            # Use default tokenizer for text fields
            schema_builder.add_text_field(
                col.replace(" ", "_").lower(),  # Sanitize field name
                stored=True,
                tokenizer_name="default"
            )
        
        # Add a combined field for full-text search
        schema_builder.add_text_field(
            "_combined",
            stored=False,
            tokenizer_name="en_stem"  # Use stemming for better recall
        )
        
        self.schema = schema_builder.build()
        return self.schema
    
    def create_index(self, columns: List[str], clear_existing: bool = True):
        """Create or open Tantivy index."""
        if clear_existing and self.index_path.exists():
            shutil.rmtree(self.index_path)
            logger.info(f"Cleared existing Tantivy index at {self.index_path}")
        
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Create schema
        self.create_schema(columns)
        
        # Create index
        self.index = tantivy.Index(self.schema, path=str(self.index_path))
        
        # Configure writer
        writer_heap = SearchConfig.TANTIVY_MEMORY_BUDGET
        self.writer = self.index.writer(writer_heap)
        
        logger.info(f"Created Tantivy index at {self.index_path}")
    
    def add_documents(self, df: pd.DataFrame, text_columns: List[str], 
                     batch_size: int = 1000, show_progress: bool = True):
        """Add documents to Tantivy index in batches."""
        if not self.writer:
            raise ValueError("Index not initialized. Call create_index first.")
        
        total_docs = len(df)
        if show_progress:
            pbar = tqdm(total=total_docs, desc="Indexing documents (Tantivy)")
        
        # Process in batches
        for start_idx in range(0, total_docs, batch_size):
            end_idx = min(start_idx + batch_size, total_docs)
            batch_df = df.iloc[start_idx:end_idx]
            
            for idx, row in batch_df.iterrows():
                # Create Tantivy document
                doc = tantivy.Document()
                
                # Add document ID
                doc.add_integer("doc_id", int(idx))
                
                # Add text fields
                combined_text = []
                for col in text_columns:
                    if col in row:
                        value = handle_edge_cases(row[col])
                        if value:
                            # Add to individual field
                            field_name = col.replace(" ", "_").lower()
                            doc.add_text(field_name, clean_text(value))
                            combined_text.append(clean_text(value))
                
                # Add combined field for cross-field search
                if combined_text:
                    doc.add_text("_combined", " ".join(combined_text))
                
                self.writer.add_document(doc)
            
            if show_progress:
                pbar.update(end_idx - start_idx)
        
        if show_progress:
            pbar.close()
    
    def commit(self):
        """Commit changes to index."""
        if self.writer:
            self.writer.commit()
            self.writer.wait_merging_threads()
            logger.info("Committed Tantivy index")
    
    def reload(self):
        """Reload the index to ensure it points to the last commit."""
        if self.index:
            self.index.reload()


class FAISSIndexer:
    """Handle FAISS index creation and management."""
    
    def __init__(self, index_path: str = SearchConfig.FAISS_INDEX_PATH):
        self.index_path = Path(index_path)
        self.index = None
        self.embedding_client = None
        self.id_map = {}  # Map FAISS index to document IDs
    
    async def initialize_embedding_client(self):
        """Initialize the embedding client."""
        self.embedding_client = OptimizedEmbeddingClient()
    
    def create_index(self, dimension: int = SearchConfig.VECTOR_DIMENSIONS):
        """Create FAISS index with L2 distance metric."""
        # Use IndexFlatL2 for exact search
        # For larger datasets, consider IndexIVFFlat or IndexHNSWFlat
        self.index = faiss.IndexFlatL2(dimension)
        logger.info(f"Created FAISS index with dimension {dimension}")
    
    async def add_embeddings(self, df: pd.DataFrame, columns: List[str], 
                            show_progress: bool = True):
        """Generate embeddings and add to FAISS index."""
        if not self.embedding_client:
            await self.initialize_embedding_client()
        
        all_embeddings = []
        all_ids = []
        
        # Process each row to create combined text from specified columns
        texts = []
        doc_ids = []
        
        for idx in df.index:
            row = df.loc[idx]
            combined_text_parts = []
            
            for col in columns:
                if col not in df.columns:
                    continue
                    
                value = row[col]
                cleaned = clean_text(handle_edge_cases(value))
                if cleaned:
                    combined_text_parts.append(cleaned)
            
            # Only add if we have some text
            if combined_text_parts:
                combined_text = " ".join(combined_text_parts)
                texts.append(combined_text)
                doc_ids.append(idx)
        
        if texts:
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} documents from columns: {columns}")
            embeddings = await self.embedding_client.get_embeddings(
                texts, 
                show_progress=show_progress
            )
            
            # Convert to float32 for FAISS
            embeddings = embeddings.astype('float32')
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            # Update ID map - map FAISS index position to document ID
            start_idx = len(self.id_map)
            for i, doc_id in enumerate(doc_ids):
                self.id_map[start_idx + i] = int(doc_id)
            
            logger.info(f"Added {len(embeddings)} embeddings to FAISS index")
            logger.info(f"ID map now contains {len(self.id_map)} mappings")
    
    def save(self):
        """Save FAISS index and metadata to disk."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = self.index_path / "index.faiss"
        faiss.write_index(self.index, str(index_file))
        
        # Save ID map
        id_map_file = self.index_path / "id_map.json"
        with open(id_map_file, 'w') as f:
            json.dump({str(k): v for k, v in self.id_map.items()}, f)
        
        logger.info(f"Saved FAISS index to {self.index_path}")
    
    def load(self):
        """Load FAISS index and metadata from disk."""
        index_file = self.index_path / "index.faiss"
        if index_file.exists():
            self.index = faiss.read_index(str(index_file))
            
            # Load ID map
            id_map_file = self.index_path / "id_map.json"
            if id_map_file.exists():
                with open(id_map_file, 'r') as f:
                    self.id_map = {int(k): v for k, v in json.load(f).items()}
            
            logger.info(f"Loaded FAISS index from {self.index_path}")
            return True
        return False


class DataIngestion:
    """Main class for data ingestion and indexing."""
    
    def __init__(self):
        self.tantivy_indexer = TantivyIndexer()
        self.faiss_indexer = FAISSIndexer()
        self.monitor = PerformanceMonitor()
        self.df = None
    
    async def ingest_data(self, 
                          data_path: str = SearchConfig.DATA_PATH,
                          reindex: bool = SearchConfig.REINDEX):
        """
        Main ingestion pipeline for creating search indices.
        
        Args:
            data_path: Path to parquet file
            reindex: Whether to rebuild indices from scratch
        """
        logger.info(f"Starting data ingestion from {data_path}")
        self.monitor.start_timer("total_ingestion")
        
        # Load data
        self.monitor.start_timer("data_loading")
        self.df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(self.df)} rows with {len(self.df.columns)} columns")
        self.monitor.end_timer("data_loading")
        
        # Validate data
        if not DataProcessor.validate_dataframe(self.df):
            raise ValueError("Invalid dataframe")
        
        # Calculate Tantivy heap size if not set
        if not SearchConfig.TANTIVY_HEAP_SIZE:
            file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
            SearchConfig.TANTIVY_HEAP_SIZE = SearchConfig.calculate_tantivy_heap_size(file_size_mb)
        
        # Get text columns for indexing
        text_columns = SearchConfig.get_search_columns(self.df)
        logger.info(f"Found {len(text_columns)} text columns for indexing")
        
        # Create Tantivy index
        if reindex or not (Path(SearchConfig.TANTIVY_INDEX_PATH) / "meta.json").exists():
            self.monitor.start_timer("tantivy_indexing")
            await self._create_tantivy_index(text_columns)
            self.monitor.end_timer("tantivy_indexing")
        else:
            logger.info("Using existing Tantivy index")
        
        # Create FAISS index
        if reindex or not self.faiss_indexer.load():
            self.monitor.start_timer("faiss_indexing")
            await self._create_faiss_index()
            self.monitor.end_timer("faiss_indexing")
        else:
            logger.info("Using existing FAISS index")
        
        self.monitor.end_timer("total_ingestion")
        self.monitor.log_metrics()
        
        logger.info("Data ingestion completed successfully")
        return True
    
    async def _create_tantivy_index(self, text_columns: List[str]):
        """Create Tantivy index for keyword search."""
        logger.info("Creating Tantivy index...")
        
        # Create index
        self.tantivy_indexer.create_index(text_columns, clear_existing=True)
        
        # Process data in chunks for memory efficiency
        chunk_size = SearchConfig.CHUNK_SIZE
        
        if len(self.df) > chunk_size * 10:  # Use multiprocessing for large datasets
            await self._parallel_tantivy_indexing(text_columns)
        else:
            self.tantivy_indexer.add_documents(
                self.df, 
                text_columns,
                batch_size=chunk_size
            )
        
        # Commit index
        self.tantivy_indexer.commit()
        self.tantivy_indexer.reload()
    
    async def _parallel_tantivy_indexing(self, text_columns: List[str]):
        """Parallel indexing for large datasets."""
        logger.info(f"Using {SearchConfig.PARALLEL_WORKERS} workers for parallel indexing")
        
        # Split dataframe into chunks
        chunk_size = len(self.df) // SearchConfig.PARALLEL_WORKERS
        chunks = [
            self.df.iloc[i:i+chunk_size] 
            for i in range(0, len(self.df), chunk_size)
        ]
        
        # Process chunks in parallel
        # Note: For simplicity, we're using sequential processing here
        # In production, you might want to use ray or dask for true parallel processing
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            self.tantivy_indexer.add_documents(
                chunk, 
                text_columns,
                batch_size=SearchConfig.CHUNK_SIZE,
                show_progress=True
            )
    
    async def _create_faiss_index(self):
        """Create FAISS index for semantic search."""
        logger.info("Creating FAISS index...")
        
        # Initialize embedding client
        await self.faiss_indexer.initialize_embedding_client()
        
        # Create index
        self.faiss_indexer.create_index()
        
        # Determine columns to use for embeddings
        columns_to_embed = SearchConfig.EMBEDDINGS_COLUMNS
        
        # Validate columns exist
        valid_columns = []
        for col in columns_to_embed:
            if col in self.df.columns:
                valid_columns.append(col)
            else:
                logger.warning(f"Column '{col}' not found in dataframe, skipping")
        
        if not valid_columns:
            # If no valid columns specified, use all text columns
            valid_columns = SearchConfig.get_search_columns(self.df)
            if valid_columns:
                logger.info(f"Using detected text columns for embeddings: {valid_columns[:5]}...")  # Show first 5
            else:
                logger.error("No text columns found for embedding generation")
                return
        
        # Add embeddings
        await self.faiss_indexer.add_embeddings(
            self.df,
            valid_columns,
            show_progress=True
        )
        
        # Save index
        self.faiss_indexer.save()


async def main():
    """Main entry point for ingestion."""
    # Validate configuration
    if not SearchConfig.validate():
        return
    
    # Create ingestion instance
    ingestion = DataIngestion()
    
    # Run ingestion
    try:
        await ingestion.ingest_data()
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    # Run ingestion
    asyncio.run(main())
