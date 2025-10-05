"""
Configuration file for JSON search system
"""

import os

# Directory paths
DATA_DIR = "json_data"  # Directory containing JSON files
INDEX_DIR = "tantivy_index"  # Directory for Tantivy index

# Indexing settings
BATCH_SIZE = 100  # Number of documents to index at once
MAX_FIELD_LENGTH = 1000000  # Maximum field length for indexing

# Search settings
DEFAULT_SEARCH_LIMIT = 10  # Default number of search results
TOP_K_RESULTS = 2  # Number of top results to return after RRF
RRF_K = 60  # Reciprocal Rank Fusion constant (typically 60)

# Search modes
SEARCH_MODES = {
    "keyword": "exact_filename",  # Exact match on filename
    "fulltext": "content_search"  # Full text search in content
}

# Field names for Tantivy index
FIELD_FILENAME = "filename"
FIELD_CONTENT = "content"
FIELD_PATH = "filepath"
FIELD_RAW_JSON = "raw_json"  # Store complete JSON

# Performance settings
NUM_THREADS = 4  # Number of threads for indexing
MEMORY_SIZE = 100_000_000  # Index writer memory size (100MB)

# Mock data generation settings (for testing)
MOCK_DATA_COUNT = 1000  # Number of mock JSON files to generate (reduced for faster testing)
MOCK_DATA_SEED = 42  # Random seed for consistent mock data