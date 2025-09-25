import json
import pandas as pd
import numpy as np
import tantivy
import tempfile
import shutil
import pickle
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Union, Any, Optional, Tuple
from itertools import permutations
import logging
from datetime import datetime, date
from decimal import Decimal
import threading
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle various data types safely."""
    
    def default(self, obj):
        """Convert non-serializable objects to serializable format."""
        try:
            # Handle pandas/numpy types
            if pd.isna(obj):
                return None
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, (datetime, date)):
                return obj.isoformat()
            elif isinstance(obj, Decimal):
                return float(obj)
            elif isinstance(obj, bytes):
                return obj.decode('utf-8', errors='ignore')
            elif isinstance(obj, set):
                return list(obj)
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                return super().default(obj)
        except Exception:
            return str(obj)


class DataCache:
    """Handles caching of DataFrame and index metadata."""
    
    def __init__(self, cache_dir: str = None):
        """Initialize cache with directory."""
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), 'search_engine_cache')
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        self.df_cache_path = os.path.join(self.cache_dir, 'dataframe.pkl')
        self.df_lower_cache_path = os.path.join(self.cache_dir, 'dataframe_lower.pkl')
        self.metadata_path = os.path.join(self.cache_dir, 'metadata.json')
        self.index_path = os.path.join(self.cache_dir, 'tantivy_index')
    
    def get_data_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of DataFrame for change detection."""
        try:
            # Create hash from shape, columns, and sample of data
            hash_obj = hashlib.sha256()
            hash_obj.update(str(df.shape).encode())
            hash_obj.update(str(list(df.columns)).encode())
            hash_obj.update(str(df.dtypes.to_dict()).encode())
            
            # Sample some rows for hash (faster than hashing entire df)
            sample_size = min(100, len(df))
            if sample_size > 0:
                sample_indices = np.linspace(0, len(df)-1, sample_size, dtype=int)
                sample_data = df.iloc[sample_indices].to_json()
                hash_obj.update(sample_data.encode())
            
            return hash_obj.hexdigest()
        except Exception as e:
            logger.warning(f"Error calculating data hash: {e}")
            return ""
    
    def is_cache_valid(self, df: pd.DataFrame) -> bool:
        """Check if cached data is valid for given DataFrame."""
        try:
            if not os.path.exists(self.metadata_path):
                return False
            
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            current_hash = self.get_data_hash(df)
            return metadata.get('data_hash') == current_hash
            
        except Exception as e:
            logger.warning(f"Error checking cache validity: {e}")
            return False
    
    def save_cache(self, df: pd.DataFrame, df_lower: pd.DataFrame, columns: List[str]):
        """Save DataFrame and metadata to cache."""
        try:
            # Save DataFrames using pickle (faster than parquet for mixed types)
            with open(self.df_cache_path, 'wb') as f:
                pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            with open(self.df_lower_cache_path, 'wb') as f:
                pickle.dump(df_lower, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save metadata
            metadata = {
                'data_hash': self.get_data_hash(df),
                'columns': columns,
                'shape': df.shape,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            logger.info(f"Cache saved to {self.cache_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
            return False
    
    def load_cache(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[List[str]]]:
        """Load cached DataFrames and metadata."""
        try:
            if not os.path.exists(self.df_cache_path):
                return None, None, None
            
            with open(self.df_cache_path, 'rb') as f:
                df = pickle.load(f)
            
            with open(self.df_lower_cache_path, 'rb') as f:
                df_lower = pickle.load(f)
            
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            columns = metadata.get('columns', [])
            
            logger.info(f"Cache loaded from {self.cache_dir}")
            return df, df_lower, columns
            
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return None, None, None
    
    def clear_cache(self):
        """Clear all cached data."""
        try:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")


class FastSearchEngine:
    """Fast search engine with persistent indexing and caching."""
    
    # Class-level cache for singleton pattern
    _instances = {}
    _lock = threading.Lock()
    
    def __new__(cls, df: pd.DataFrame = None, 
                cache_dir: str = None, 
                force_rebuild: bool = False,
                use_cache: bool = True):
        """
        Implement singleton pattern with cache key.
        Returns existing instance if data hasn't changed.
        """
        # Create cache key from cache directory
        cache_key = cache_dir or 'default'
        
        with cls._lock:
            # If force rebuild, remove existing instance
            if force_rebuild and cache_key in cls._instances:
                old_instance = cls._instances[cache_key]
                if hasattr(old_instance, '_cleanup'):
                    old_instance._cleanup()
                del cls._instances[cache_key]
            
            # Return existing instance if available and valid
            if cache_key in cls._instances and use_cache:
                instance = cls._instances[cache_key]
                if df is not None:
                    # Check if data has changed
                    if instance.cache.is_cache_valid(df):
                        logger.info("Using existing search engine instance (data unchanged)")
                        return instance
                    else:
                        logger.info("Data has changed, rebuilding index...")
                        if hasattr(instance, '_cleanup'):
                            instance._cleanup()
                        del cls._instances[cache_key]
                else:
                    # No new data provided, return existing instance
                    return instance
            
            # Create new instance
            instance = super().__new__(cls)
            cls._instances[cache_key] = instance
            return instance
    
    def __init__(self, df: pd.DataFrame = None, 
                 cache_dir: str = None,
                 force_rebuild: bool = False,
                 use_cache: bool = True):
        """
        Initialize the search engine with a DataFrame.
        
        Args:
            df: Input pandas DataFrame (optional if using cache)
            cache_dir: Directory for cache and index storage
            force_rebuild: Force rebuild of index even if cache exists
            use_cache: Whether to use caching (default True)
        """
        # Skip initialization if already initialized
        if hasattr(self, '_initialized'):
            return
        
        self.use_cache = use_cache
        self.cache = DataCache(cache_dir)
        self.tantivy_index = None
        self.schema = None
        self.searcher = None
        self._initialized = False
        
        # Try to load from cache first
        if use_cache and not force_rebuild and df is None:
            loaded = self._load_from_cache()
            if loaded:
                self._initialized = True
                return
        
        # Require DataFrame if not loaded from cache
        if df is None:
            raise ValueError("DataFrame required when cache is empty or disabled")
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        
        # Initialize with new data
        self._initialize_with_data(df, force_rebuild)
        self._initialized = True
    
    def _initialize_with_data(self, df: pd.DataFrame, force_rebuild: bool = False):
        """Initialize engine with new DataFrame."""
        try:
            # Check if we can use existing index
            if self.use_cache and not force_rebuild and self.cache.is_cache_valid(df):
                loaded = self._load_from_cache()
                if loaded:
                    logger.info("Loaded existing index and cache")
                    return
            
            # Process new data
            self.df = df.copy()
            self.columns = list(df.columns)
            
            # Clean the DataFrame
            self._clean_dataframe()
            
            # Pre-convert string columns to lowercase for faster searching
            self.df_lower = self.df.copy()
            for col in self.df_lower.select_dtypes(include=['object', 'string']).columns:
                self.df_lower[col] = self.df_lower[col].fillna('').astype(str).str.lower()
            
            # Save cache before building index
            if self.use_cache:
                self.cache.save_cache(self.df, self.df_lower, self.columns)
            
            # Initialize tantivy index
            self._init_tantivy_index(force_rebuild)
            
        except Exception as e:
            logger.error(f"Error initializing with data: {e}")
            raise
    
    def _load_from_cache(self) -> bool:
        """Load data from cache."""
        try:
            df, df_lower, columns = self.cache.load_cache()
            
            if df is None:
                return False
            
            self.df = df
            self.df_lower = df_lower
            self.columns = columns
            
            # Load existing tantivy index
            if os.path.exists(self.cache.index_path):
                self._load_tantivy_index()
                return True
            else:
                # Cache exists but index doesn't, rebuild index
                logger.info("Cache found but index missing, rebuilding index...")
                self._init_tantivy_index(force_rebuild=True)
                return True
                
        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
            return False
    
    def _load_tantivy_index(self):
        """Load existing tantivy index from disk."""
        try:
            # Load schema
            schema_builder = tantivy.SchemaBuilder()
            
            # Add doc_id field
            schema_builder.add_integer_field("doc_id", stored=True, indexed=True)
            
            # Add text fields for each column
            for col in self.columns:
                schema_builder.add_text_field(
                    col, 
                    stored=True,
                    tokenizer_name="en_stem"
                )
            
            # Add combined field
            schema_builder.add_text_field(
                "_all_text",
                stored=False,
                tokenizer_name="en_stem"
            )
            
            self.schema = schema_builder.build()
            
            # Open existing index
            self.tantivy_index = tantivy.Index(self.schema, path=str(self.cache.index_path))
            self.tantivy_index.reload()
            self.searcher = self.tantivy_index.searcher()
            
            logger.info(f"Loaded existing tantivy index from {self.cache.index_path}")
            
        except Exception as e:
            logger.error(f"Error loading tantivy index: {e}")
            # If loading fails, rebuild
            self._init_tantivy_index(force_rebuild=True)
    
    def _clean_dataframe(self):
        """Clean DataFrame by handling empty values and problematic data."""
        try:
            # Replace various empty values with empty string for string columns
            string_cols = self.df.select_dtypes(include=['object', 'string']).columns
            
            for col in string_cols:
                # Replace None, NaN, empty lists, etc. with empty string
                self.df[col] = self.df[col].apply(self._clean_value)
            
            # For numeric columns, keep NaN as is (pandas handles it well)
            # But ensure no infinity values
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                self.df[col] = self.df[col].replace([np.inf, -np.inf], np.nan)
            
        except Exception as e:
            logger.warning(f"Error during DataFrame cleaning: {e}")
    
    @staticmethod
    @lru_cache(maxsize=10000)
    def _clean_value_cached(value_str: str) -> str:
        """Cached version of clean value for common strings."""
        if value_str.lower() in ['nan', 'none', 'null', 'na', 'n/a', '<na>']:
            return ''
        return value_str
    
    def _clean_value(self, value):
        """Clean individual cell values."""
        try:
            # Handle None, NaN
            if pd.isna(value):
                return ''
            
            # Handle empty containers
            if isinstance(value, (list, tuple, dict, set)):
                if not value:
                    return ''
                return str(value)
            
            # Handle bytes
            if isinstance(value, bytes):
                return value.decode('utf-8', errors='ignore')
            
            # Convert to string and strip whitespace
            str_val = str(value).strip()
            
            # Use cached cleaning for common values
            return self._clean_value_cached(str_val)
            
        except Exception:
            return ''
    
    def _init_tantivy_index(self, force_rebuild: bool = False):
        """Initialize tantivy index for full-text search."""
        try:
            index_path = self.cache.index_path
            
            # Check if index already exists and is valid
            if not force_rebuild and os.path.exists(index_path):
                try:
                    self._load_tantivy_index()
                    return
                except Exception as e:
                    logger.warning(f"Failed to load existing index: {e}")
            
            # Clear old index if exists
            if os.path.exists(index_path):
                shutil.rmtree(index_path)
            
            Path(index_path).mkdir(parents=True, exist_ok=True)
            
            # Build schema
            schema_builder = tantivy.SchemaBuilder()
            
            # Add doc_id field for referencing
            schema_builder.add_integer_field("doc_id", stored=True, indexed=True)
            
            # Add text fields for each column
            for col in self.columns:
                schema_builder.add_text_field(
                    col, 
                    stored=True,
                    tokenizer_name="en_stem"  # Use stemming for better recall
                )
            
            # Add a combined field for better search
            schema_builder.add_text_field(
                "_all_text",
                stored=False,
                tokenizer_name="en_stem"
            )
            
            self.schema = schema_builder.build()
            
            # Create index
            self.tantivy_index = tantivy.Index(self.schema, path=str(index_path))
            
            # Index documents
            writer = self.tantivy_index.writer(heap_size=100_000_000)  # 100MB heap
            
            # Batch processing for better performance
            batch_size = 1000
            for batch_start in range(0, len(self.df), batch_size):
                batch_end = min(batch_start + batch_size, len(self.df))
                
                for idx in range(batch_start, batch_end):
                    try:
                        row = self.df.iloc[idx]
                        doc = tantivy.Document()
                        doc.add_integer("doc_id", int(idx))
                        
                        all_text_parts = []
                        
                        for col in self.columns:
                            value = row[col]
                            
                            # Clean and convert value
                            if pd.isna(value):
                                text_value = ""
                            elif isinstance(value, (list, dict, tuple, set)):
                                text_value = str(value) if value else ""
                            else:
                                text_value = str(value).strip()
                            
                            # Only add non-empty values
                            if text_value and text_value.lower() not in ['nan', 'none', 'null']:
                                doc.add_text(col, text_value)
                                all_text_parts.append(text_value)
                        
                        # Add combined text field if there's any content
                        if all_text_parts:
                            doc.add_text("_all_text", " ".join(all_text_parts))
                        
                        writer.add_document(doc)
                        
                    except Exception as e:
                        logger.warning(f"Error indexing row {idx}: {e}")
                        continue
                
                # Commit batch
                if batch_end % 5000 == 0:
                    writer.commit()
                    logger.info(f"Indexed {batch_end} documents...")
            
            # Final commit
            writer.commit()
            writer.wait_merging_threads()
            
            # Reload index and create searcher
            self.tantivy_index.reload()
            self.searcher = self.tantivy_index.searcher()
            
            logger.info(f"Tantivy index created at {index_path}")
            
        except Exception as e:
            logger.error(f"Error initializing tantivy index: {e}")
            raise RuntimeError(f"Failed to initialize search index: {e}")
    
    def _apply_filters_sequence(self, filters: Dict[str, str], sequence: List[str]) -> pd.DataFrame:
        """
        Apply filters in a specific sequence with empty value handling.
        
        Args:
            filters: Dictionary of column:value filters
            sequence: Order in which to apply filters
            
        Returns:
            Filtered DataFrame
        """
        result = self.df_lower.copy()
        
        for key in sequence:
            if key in filters:
                filter_value = str(filters[key]).strip().lower()
                
                # Skip empty filter values
                if not filter_value or filter_value in ['nan', 'none', 'null']:
                    continue
                
                # Check column exists
                if key not in result.columns:
                    logger.warning(f"Column '{key}' not found in DataFrame")
                    continue
                
                # Apply filter with proper null handling
                try:
                    # Fill NaN with empty string for comparison
                    column_data = result[key].fillna('').astype(str)
                    mask = column_data.str.contains(
                        filter_value, 
                        case=False, 
                        na=False, 
                        regex=False
                    )
                    result = result[mask]
                    
                except Exception as e:
                    logger.warning(f"Error applying filter on column '{key}': {e}")
                    continue
                
                if result.empty:
                    break
        
        return self.df.loc[result.index] if not result.empty else pd.DataFrame()
    
    def _pandas_search(self, filters: Dict[str, str]) -> Dict[str, Any]:
        """
        Perform pandas-based filtering with multiple sequence attempts.
        
        Args:
            filters: Dictionary of column:value filters
            
        Returns:
            Dictionary with results and count
        """
        try:
            if not filters:
                raise ValueError("Filters dictionary cannot be empty for pandas mode")
            
            # Clean filters - remove empty values
            cleaned_filters = {}
            for k, v in filters.items():
                if v is not None:
                    v_str = str(v).strip()
                    if v_str and v_str.lower() not in ['nan', 'none', 'null', '']:
                        cleaned_filters[k] = v_str
            
            if not cleaned_filters:
                raise ValueError("All filter values are empty or invalid")
            
            # Validate filter keys
            invalid_keys = set(cleaned_filters.keys()) - set(self.columns)
            if invalid_keys:
                logger.warning(f"Invalid filter keys will be ignored: {invalid_keys}")
                cleaned_filters = {k: v for k, v in cleaned_filters.items() 
                                 if k in self.columns}
            
            if not cleaned_filters:
                raise ValueError("No valid filters remaining after validation")
            
            # Try original sequence first
            original_sequence = list(cleaned_filters.keys())
            result = self._apply_filters_sequence(cleaned_filters, original_sequence)
            
            if not result.empty:
                return self._format_results(result)
            
            # Try up to 3 different permutations
            max_attempts = min(3, len(list(permutations(cleaned_filters.keys()))))
            attempt = 0
            
            for sequence in permutations(cleaned_filters.keys()):
                if attempt >= max_attempts:
                    break
                
                if list(sequence) != original_sequence:  # Skip original
                    result = self._apply_filters_sequence(cleaned_filters, list(sequence))
                    if not result.empty:
                        return self._format_results(result)
                    attempt += 1
            
            # If no results after attempts, fall back to best_match
            logger.info("No results with pandas filters, falling back to best_match")
            query = " ".join(cleaned_filters.values())
            return self._best_match_search(query)
            
        except Exception as e:
            logger.error(f"Error in pandas search: {e}")
            return {"results": [], "count": 0, "error": str(e)}
    
    def _format_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Format DataFrame results to dictionary with proper handling of special values.
        
        Args:
            df: Results DataFrame
            
        Returns:
            Formatted dictionary
        """
        try:
            # Replace NaN and infinity with None for JSON serialization
            df_clean = df.copy()
            
            # Handle numeric columns
            for col in df_clean.select_dtypes(include=[np.number]).columns:
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
            
            # Convert to records
            records = df_clean.to_dict('records')
            
            # Clean each record
            cleaned_records = []
            for record in records:
                cleaned_record = {}
                for key, value in record.items():
                    # Handle NaN/None
                    if pd.isna(value):
                        cleaned_record[key] = None
                    # Handle empty strings
                    elif isinstance(value, str) and not value.strip():
                        cleaned_record[key] = ""
                    else:
                        cleaned_record[key] = value
                cleaned_records.append(cleaned_record)
            
            return {
                "results": cleaned_records,
                "count": len(cleaned_records)
            }
            
        except Exception as e:
            logger.error(f"Error formatting results: {e}")
            return {"results": [], "count": 0, "error": str(e)}
    
    def _calculate_rrf_score(self, ranks: List[int], k: int = 60) -> float:
        """
        Calculate Reciprocal Rank Fusion score.
        
        Args:
            ranks: List of ranks from different searches
            k: Constant for RRF (default 60)
            
        Returns:
            RRF score
        """
        return sum(1.0 / (k + rank) for rank in ranks)
    
    def _best_match_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Perform best_match search using tantivy with RRF fusion.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            Dictionary with results and count
        """
        try:
            if not query or not query.strip():
                raise ValueError("Query string cannot be empty for best_match mode")
            
            # Clean query
            query_clean = query.strip()
            if query_clean.lower() in ['nan', 'none', 'null']:
                raise ValueError("Invalid query string")
            
            # Store results from each column search
            column_results = {}
            
            # First try searching in the combined field
            try:
                parsed_query = self.tantivy_index.parse_query(query_clean, ["_all_text"])
                search_result = self.searcher.search(parsed_query, top_k)
                
                if search_result.count > 0:
                    # Get results from combined field search
                    combined_results = []
                    for rank, (score, doc_address) in enumerate(search_result.hits, 1):
                        doc = self.searcher.doc(doc_address)
                        doc_id = doc.get_first("doc_id")
                        if doc_id is not None:
                            combined_results.append({
                                'doc_id': doc_id,
                                'rank': rank,
                                'score': score * 1.5  # Boost combined field results
                            })
                    column_results['_all'] = combined_results
            except Exception as e:
                logger.warning(f"Combined field search failed: {e}")
            
            # Search in each column separately for RRF
            for col in self.columns:
                try:
                    parsed_query = self.tantivy_index.parse_query(query_clean, [col])
                    search_result = self.searcher.search(parsed_query, top_k * 2)
                    
                    # Store results with doc_ids and scores
                    column_results[col] = []
                    for rank, (score, doc_address) in enumerate(search_result.hits, 1):
                        doc = self.searcher.doc(doc_address)
                        doc_id = doc.get_first("doc_id")
                        if doc_id is not None:
                            column_results[col].append({
                                'doc_id': doc_id,
                                'rank': rank,
                                'score': score
                            })
                except Exception as e:
                    logger.debug(f"Search failed for column {col}: {e}")
                    continue
            
            if not column_results:
                return {"results": [], "count": 0}
            
            # Combine results using RRF
            doc_rrf_scores = {}
            
            for col, results in column_results.items():
                for result in results:
                    doc_id = result['doc_id']
                    rank = result['rank']
                    
                    if doc_id not in doc_rrf_scores:
                        doc_rrf_scores[doc_id] = {'ranks': [], 'columns': []}
                    
                    doc_rrf_scores[doc_id]['ranks'].append(rank)
                    if col != '_all':
                        doc_rrf_scores[doc_id]['columns'].append(col)
            
            # Calculate final RRF scores
            final_scores = []
            for doc_id, info in doc_rrf_scores.items():
                rrf_score = self._calculate_rrf_score(info['ranks'])
                final_scores.append((doc_id, rrf_score, info['columns']))
            
            # Sort by RRF score descending
            final_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get top k results
            top_results = []
            for doc_id, rrf_score, matched_columns in final_scores[:top_k]:
                try:
                    row_dict = self.df.iloc[doc_id].to_dict()
                    
                    # Clean the row data
                    cleaned_row = {}
                    for key, value in row_dict.items():
                        if pd.isna(value):
                            cleaned_row[key] = None
                        elif isinstance(value, str) and not value.strip():
                            cleaned_row[key] = ""
                        else:
                            cleaned_row[key] = value
                    
                    cleaned_row['_score'] = float(rrf_score)
                    cleaned_row['_matched_columns'] = list(set(matched_columns))
                    top_results.append(cleaned_row)
                    
                except Exception as e:
                    logger.warning(f"Error processing result for doc_id {doc_id}: {e}")
                    continue
            
            return {
                "results": top_results,
                "count": len(top_results)
            }
            
        except Exception as e:
            logger.error(f"Error in best_match search: {e}")
            return {"results": [], "count": 0, "error": str(e)}
    
    def search(self, 
               mode: str = "pandas",
               filters: Optional[Dict[str, str]] = None,
               query: Optional[str] = None) -> str:
        """
        Main search function supporting both pandas and best_match modes.
        
        Args:
            mode: Either "pandas" or "best_match"
            filters: Dictionary of filters for pandas mode
            query: Query string for best_match mode
            
        Returns:
            JSON string with results and count
        """
        try:
            # Ensure engine is initialized
            if not hasattr(self, '_initialized') or not self._initialized:
                raise RuntimeError("Search engine not properly initialized")
            
            # Validate mode
            if mode not in ["pandas", "best_match"]:
                raise ValueError(f"Invalid mode: {mode}. Must be 'pandas' or 'best_match'")
            
            # Execute search based on mode
            if mode == "pandas":
                if filters is None:
                    raise ValueError("Filters required for pandas mode")
                result = self._pandas_search(filters)
            
            elif mode == "best_match":
                if query is None:
                    raise ValueError("Query string required for best_match mode")
                result = self._best_match_search(query)
            
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                result = {"results": [], "count": 0, "error": "Invalid result format"}
            
            # Add metadata
            result['mode'] = mode
            result['timestamp'] = datetime.now().isoformat()
            result['cached'] = self.use_cache
            
            # Convert to JSON string with custom encoder
            return json.dumps(result, cls=SafeJSONEncoder, ensure_ascii=False, indent=None)
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            error_result = {
                "results": [],
                "count": 0,
                "error": str(e),
                "mode": mode,
                "timestamp": datetime.now().isoformat()
            }
            return json.dumps(error_result, cls=SafeJSONEncoder, ensure_ascii=False)
    
    def update_data(self, df: pd.DataFrame):
        """
        Update the search engine with new data.
        
        Args:
            df: New DataFrame to index
        """
        try:
            logger.info("Updating search engine with new data...")
            self._initialize_with_data(df, force_rebuild=True)
            logger.info("Update complete")
        except Exception as e:
            logger.error(f"Error updating data: {e}")
            raise
    
    def clear_cache(self):
        """Clear all cached data and indexes."""
        try:
            self.cache.clear_cache()
            # Remove from class instances
            cache_key = self.cache.cache_dir or 'default'
            if cache_key in self._instances:
                del self._instances[cache_key]
            logger.info("Cache and index cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the search engine."""
        try:
            stats = {
                "total_documents": len(self.df),
                "columns": self.columns,
                "cache_enabled": self.use_cache,
                "cache_directory": self.cache.cache_dir,
                "index_size_mb": 0
            }
            
            # Calculate index size
            if os.path.exists(self.cache.index_path):
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(self.cache.index_path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        total_size += os.path.getsize(fp)
                stats["index_size_mb"] = round(total_size / (1024 * 1024), 2)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def _cleanup(self):
        """Internal cleanup method."""
        # No need to delete cache on cleanup, it's persistent
        pass
    
    def __del__(self):
        """Cleanup when object is deleted."""
        # Don't delete persistent cache
        pass


# Convenience function for quick initialization
def create_search_engine(df: pd.DataFrame = None, 
                        cache_dir: str = None,
                        force_rebuild: bool = False) -> FastSearchEngine:
    """
    Create or get existing search engine instance.
    
    Args:
        df: DataFrame to index (optional if using cache)
        cache_dir: Directory for persistent storage
        force_rebuild: Force rebuild even if cache exists
    
    Returns:
        FastSearchEngine instance
    """
    return FastSearchEngine(df, cache_dir, force_rebuild)


# Example usage and testing
def example_usage():
    """Example demonstrating the search engine usage with caching."""
    
    import time
    
    print("=" * 60)
    print("SEARCH ENGINE WITH CACHING DEMO")
    print("=" * 60)
    
    # Create sample data
    print("\n1. Creating sample dataset...")
    data = {
        'id': range(5000),
        'name': [f'Person_{i}' if i % 10 != 0 else None for i in range(5000)],
        'description': [f'Description {i}' if i % 5 != 0 else '' for i in range(5000)],
        'category': [f'Cat_{i%20}' if i % 7 != 0 else np.nan for i in range(5000)],
        'score': [i * 0.1 if i % 3 != 0 else np.nan for i in range(5000)],
        'tags': [f'tag{i%10},tag{i%20}' if i % 8 != 0 else None for i in range(5000)],
    }
    
    # Add more columns
    for i in range(9):
        data[f'col_{i}'] = [
            f'value_{i}_{j}' if j % (i+2) != 0 else ''
            for j in range(5000)
        ]
    
    # Add specific test values
    data['name'][0] = 'Python Developer'
    data['description'][0] = 'Expert in machine learning'
    data['category'][0] = 'Technology'
    
    df = pd.DataFrame(data)
    print(f"DataFrame shape: {df.shape}")
    
    # First initialization (builds index)
    print("\n2. First initialization (building index)...")
    cache_dir = "./search_cache"
    
    start = time.time()
    engine = create_search_engine(df, cache_dir=cache_dir)
    init_time = (time.time() - start) * 1000
    print(f"First initialization time: {init_time:.2f}ms")
    
    # Get stats
    stats = engine.get_stats()
    print(f"Index size: {stats['index_size_mb']} MB")
    
    # Test search
    print("\n3. Testing search...")
    start = time.time()
    result = engine.search(mode="best_match", query="Python machine learning")
    search_time = (time.time() - start) * 1000
    result_dict = json.loads(result)
    print(f"Search time: {search_time:.2f}ms")
    print(f"Results found: {result_dict['count']}")
    
    # Second initialization (uses cache)
    print("\n4. Second initialization (using cache)...")
    del engine  # Delete first instance
    
    start = time.time()
    engine2 = create_search_engine(cache_dir=cache_dir)  # No DataFrame needed!
    init_time = (time.time() - start) * 1000
    print(f"Second initialization time: {init_time:.2f}ms")
    
    # Test search on cached engine
    start = time.time()
    result = engine2.search(mode="best_match", query="Python machine learning")
    search_time = (time.time() - start) * 1000
    result_dict = json.loads(result)
    print(f"Search time on cached: {search_time:.2f}ms")
    print(f"Results found: {result_dict['count']}")
    
    # Test with changed data
    print("\n5. Testing with modified data...")
    df_modified = df.copy()
    df_modified.loc[100, 'name'] = 'Modified Entry'
    
    start = time.time()
    engine3 = create_search_engine(df_modified, cache_dir=cache_dir)
    init_time = (time.time() - start) * 1000
    print(f"Rebuild time with changed data: {init_time:.2f}ms")
    
    # Test pandas mode
    print("\n6. Testing pandas mode...")
    filters = {'name': 'Python', 'description': 'machine'}
    
    start = time.time()
    result = engine3.search(mode="pandas", filters=filters)
    search_time = (time.time() - start) * 1000
    result_dict = json.loads(result)
    print(f"Pandas search time: {search_time:.2f}ms")
    print(f"Results found: {result_dict['count']}")
    
    # Force rebuild
    print("\n7. Testing force rebuild...")
    start = time.time()
    engine4 = create_search_engine(df, cache_dir=cache_dir, force_rebuild=True)
    rebuild_time = (time.time() - start) * 1000
    print(f"Force rebuild time: {rebuild_time:.2f}ms")
    
    print("\n✅ All tests completed successfully!")
    print(f"Cache directory: {cache_dir}")
    print("Note: Cache persists between runs for instant loading!")


if __name__ == "__main__":
    example_usage()
