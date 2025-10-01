#!/usr/bin/env python3
"""
Main engine script for the Advanced Search System.
Provides command-line interface for ingestion, search, and benchmarking.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Import our modules
from generate_mock_data import generate_mock_data
from ingestion import DataIngestionEngine
from search import SearchEngine
from benchmark import SearchBenchmark

# Load environment variables
load_dotenv()

class AdvancedSearchEngine:
    """Main orchestrator for the search engine system."""

    def __init__(self):
        """Initialize the engine with configuration from environment."""
        self.config = self._load_config()
        self.search_engine = None
        self.ingestion_engine = None

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        return {
            'data_path': os.getenv('DATA_PATH', 'mock_data_small.parquet'),
            'index_path': os.getenv('INDEX_PATH', './indices'),
            'embedding_columns': os.getenv('EMBEDDING_COLUMNS', 'title,description').split(','),
            'vector_dimension': int(os.getenv('VECTOR_DIMENSION', '4096')),
            'use_real_embeddings': os.getenv('USE_REAL_EMBEDDINGS', 'false').lower() == 'true',
            'openai_key': os.getenv('OPENAI_API_KEY'),
            'openai_url': os.getenv('OPENAI_API_URL'),
            'openai_model': os.getenv('OPENAI_MODEL'),
            're_index': os.getenv('RE_INDEX', 'false').lower() == 'true',
            'chunk_size': int(os.getenv('CHUNK_SIZE', '1000')),
            'default_top_k': int(os.getenv('DEFAULT_TOP_K', '5')),
            'benchmark_queries': int(os.getenv('BENCHMARK_QUERIES', '100'))
        }

    def generate_data(self, num_rows: int = 100000, output_path: str = None):
        """Generate mock data for testing."""
        if output_path is None:
            output_path = self.config['data_path']

        print(f"\nGenerating {num_rows:,} rows of mock data...")
        print(f"Output path: {output_path}")

        start_time = time.time()
        generate_mock_data(num_rows, output_path)
        elapsed = time.time() - start_time

        print(f"\nData generation completed in {elapsed:.2f} seconds")

    def ingest_data(self):
        """Run the ingestion and indexing pipeline."""
        print("\n" + "="*60)
        print("DATA INGESTION AND INDEXING")
        print("="*60)

        # Check if data file exists
        if not os.path.exists(self.config['data_path']):
            print(f"Error: Data file not found at {self.config['data_path']}")
            print("Please generate data first using: python engine.py generate")
            return False

        # Initialize ingestion engine
        self.ingestion_engine = DataIngestionEngine(
            data_path=self.config['data_path'],
            index_path=self.config['index_path'],
            embedding_columns=self.config['embedding_columns'],
            vector_dimension=self.config['vector_dimension'],
            use_real_embeddings=self.config['use_real_embeddings'],
            openai_key=self.config['openai_key'],
            openai_url=self.config['openai_url'],
            openai_model=self.config['openai_model'],
            re_index=self.config['re_index'],
            chunk_size=self.config['chunk_size']
        )

        # Run ingestion
        start_time = time.time()
        metadata = self.ingestion_engine.ingest_and_index()
        elapsed = time.time() - start_time

        print(f"\nIngestion completed in {elapsed:.2f} seconds")
        print(f"Total documents: {metadata['total_rows']:,}")

        return True

    def initialize_search(self):
        """Initialize the search engine."""
        if not os.path.exists(self.config['index_path']):
            print("Error: Index not found. Please run ingestion first.")
            return False

        self.search_engine = SearchEngine(self.config['index_path'])
        return True

    def search_interactive(self):
        """Interactive search mode."""
        if not self.initialize_search():
            return

        print("\n" + "="*60)
        print("INTERACTIVE SEARCH MODE")
        print("="*60)
        print("\nAvailable search modes:")
        print("  1. dataframe - Filter by column values")
        print("  2. keyword   - Full-text search")
        print("  3. vector    - Vector similarity search")
        print("  4. hybrid    - Combined search")
        print("\nType 'quit' to exit\n")

        while True:
            try:
                # Get search mode
                mode = input("Enter search mode (1-4): ").strip()

                if mode.lower() == 'quit':
                    break

                mode_map = {
                    '1': 'dataframe',
                    '2': 'keyword',
                    '3': 'vector',
                    '4': 'hybrid',
                    'dataframe': 'dataframe',
                    'keyword': 'keyword',
                    'vector': 'vector',
                    'hybrid': 'hybrid'
                }

                if mode not in mode_map:
                    print("Invalid mode. Please try again.")
                    continue

                search_mode = mode_map[mode]
                search_params = {}

                # Get search parameters based on mode
                if search_mode in ['dataframe', 'hybrid']:
                    print("\nEnter column-value pairs (format: column=value)")
                    print("Press Enter with empty input when done")
                    column_value_pairs = {}

                    while True:
                        pair = input("  > ").strip()
                        if not pair:
                            break

                        if '=' in pair:
                            col, val = pair.split('=', 1)
                            col = col.strip()
                            val = val.strip()

                            # Try to parse value type
                            if val.lower() == 'true':
                                val = True
                            elif val.lower() == 'false':
                                val = False
                            elif val.lower() in ['null', 'none']:
                                val = None
                            else:
                                try:
                                    val = int(val)
                                except ValueError:
                                    try:
                                        val = float(val)
                                    except ValueError:
                                        pass  # Keep as string

                            column_value_pairs[col] = val

                    if column_value_pairs:
                        search_params['column_value_pairs'] = column_value_pairs

                if search_mode in ['keyword', 'hybrid']:
                    keywords_input = input("\nEnter keywords (space-separated): ").strip()
                    if keywords_input:
                        search_params['keywords'] = keywords_input.split()

                if search_mode in ['vector', 'hybrid']:
                    print("\nUsing random vector for demo...")
                    search_params['query_vector'] = np.random.randn(
                        self.search_engine.vector_dimension
                    ).astype(np.float32)

                # Execute search
                print("\nSearching...")
                result = self.search_engine.search(
                    search_mode=search_mode,
                    top_k=self.config['default_top_k'],
                    **search_params
                )

                # Display results
                print(f"\nStatus: {result['status']}")
                print(f"Results found: {result['count']}")
                print(f"Response time: {result['time_taken']*1000:.2f}ms")

                if result['results']:
                    print("\nTop results:")
                    for i, doc in enumerate(result['results'][:5], 1):
                        print(f"\n--- Result {i} ---")
                        # Show key fields
                        for key in ['id', 'title', 'status', 'type', 'category']:
                            if key in doc:
                                value = str(doc[key])[:100]  # Truncate long values
                                print(f"  {key}: {value}")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

    def run_benchmark(self):
        """Run performance benchmarks."""
        if not os.path.exists(self.config['index_path']):
            print("Error: Index not found. Please run ingestion first.")
            return

        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK")
        print("="*60)

        benchmark = SearchBenchmark(self.config['index_path'])
        stats = benchmark.run_comprehensive_benchmark(
            num_queries_per_mode=self.config['benchmark_queries']
        )

        return stats

    def show_status(self):
        """Show system status and configuration."""
        print("\n" + "="*60)
        print("ADVANCED SEARCH ENGINE STATUS")
        print("="*60)

        print("\nConfiguration:")
        print(f"  Data Path: {self.config['data_path']}")
        print(f"  Index Path: {self.config['index_path']}")
        print(f"  Embedding Columns: {', '.join(self.config['embedding_columns'])}")
        print(f"  Vector Dimension: {self.config['vector_dimension']}")
        print(f"  Use Real Embeddings: {self.config['use_real_embeddings']}")

        # Check data file
        data_exists = os.path.exists(self.config['data_path'])
        print(f"\nData File: {'✓ Found' if data_exists else '✗ Not Found'}")

        if data_exists:
            file_size = os.path.getsize(self.config['data_path']) / (1024**3)
            print(f"  Size: {file_size:.2f} GB")

        # Check indices
        index_exists = os.path.exists(self.config['index_path'])
        print(f"\nIndices: {'✓ Found' if index_exists else '✗ Not Found'}")

        if index_exists:
            metadata_path = Path(self.config['index_path']) / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"  Documents: {metadata['total_rows']:,}")
                print(f"  Columns: {metadata['total_columns']}")

            # Check individual indices
            indices = {
                'DuckDB': Path(self.config['index_path']) / 'duckdb_index.db',
                'Tantivy': Path(self.config['index_path']) / 'tantivy_index',
                'FAISS': Path(self.config['index_path']) / 'faiss_indices'
            }

            for name, path in indices.items():
                exists = path.exists()
                print(f"  {name}: {'✓' if exists else '✗'}")

def print_usage():
    """Print usage information."""
    print("""
Usage: python engine.py [command] [options]

Commands:
  generate    Generate mock data
              Options:
                --rows NUM     Number of rows (default: 100000)
                --output PATH  Output file path

  ingest      Ingest and index data
              Options:
                --reindex      Force re-indexing

  search      Interactive search mode

  benchmark   Run performance benchmarks
              Options:
                --queries NUM  Queries per mode (default: 100)

  status      Show system status

Examples:
  python engine.py generate --rows 1000000
  python engine.py ingest
  python engine.py search
  python engine.py benchmark --queries 200
  python engine.py status
""")

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1].lower()
    engine = AdvancedSearchEngine()

    if command == 'generate':
        # Parse options
        rows = 100000
        output = None

        for i in range(2, len(sys.argv)):
            if sys.argv[i] == '--rows' and i + 1 < len(sys.argv):
                rows = int(sys.argv[i + 1])
            elif sys.argv[i] == '--output' and i + 1 < len(sys.argv):
                output = sys.argv[i + 1]

        engine.generate_data(rows, output)

    elif command == 'ingest':
        # Check for reindex flag
        if '--reindex' in sys.argv:
            os.environ['RE_INDEX'] = 'true'
            engine.config['re_index'] = True

        engine.ingest_data()

    elif command == 'search':
        engine.search_interactive()

    elif command == 'benchmark':
        # Parse options
        queries = 100
        for i in range(2, len(sys.argv)):
            if sys.argv[i] == '--queries' and i + 1 < len(sys.argv):
                queries = int(sys.argv[i + 1])

        engine.config['benchmark_queries'] = queries
        engine.run_benchmark()

    elif command == 'status':
        engine.show_status()

    else:
        print(f"Unknown command: {command}")
        print_usage()

if __name__ == "__main__":
    main()