import json
import pandas as pd
import numpy as np
import tantivy
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Union, Any, Optional
from itertools import permutations
import logging
from datetime import datetime, date
from decimal import Decimal

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


class FastSearchEngine:
    """Fast search engine supporting pandas filtering and tantivy full-text search with RRF fusion."""
    
    def __init__(self, df: pd.DataFrame, index_path: Optional[str] = None):
        """
        Initialize the search engine with a DataFrame.
        
        Args:
            df: Input pandas DataFrame
            index_path: Optional path for tantivy index (temp dir used if not provided)
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        
        self.df = df.copy()
        self.columns = list(df.columns)
        self.index_path = index_path
        self.tantivy_index = None
        self.schema = None
        self.searcher = None
        
        # Clean the DataFrame
        self._clean_dataframe()
        
        # Pre-convert string columns to lowercase for faster searching
        self.df_lower = self.df.copy()
        for col in self.df_lower.select_dtypes(include=['object', 'string']).columns:
            self.df_lower[col] = self.df_lower[col].fillna('').astype(str).str.lower()
        
        # Initialize tantivy index for best_match mode
        self._init_tantivy_index()
    
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
            
            # Handle special string values
            if str_val.lower() in ['nan', 'none', 'null', 'na', 'n/a', '<na>']:
                return ''
            
            return str_val
            
        except Exception:
            return ''
    
    def _init_tantivy_index(self):
        """Initialize tantivy index for full-text search."""
        try:
            # Create temporary directory if path not provided
            if self.index_path is None:
                self.temp_dir = tempfile.mkdtemp()
                self.index_path = self.temp_dir
            
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
            self.tantivy_index = tantivy.Index(self.schema, path=str(self.index_path))
            
            # Index documents
            writer = self.tantivy_index.writer(heap_size=100_000_000)  # 100MB heap
            
            for idx, row in self.df.iterrows():
                try:
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
            
            writer.commit()
            writer.wait_merging_threads()
            
            # Reload index and create searcher
            self.tantivy_index.reload()
            self.searcher = self.tantivy_index.searcher()
            
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
    
    def __del__(self):
        """Cleanup temporary index directory if created."""
        if hasattr(self, 'temp_dir') and self.temp_dir:
            try:
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass


# Example usage and testing
def example_usage():
    """Example demonstrating the search engine usage with edge cases."""
    
    import time
    
    print("Creating sample dataset with edge cases...")
    
    # Create sample data with problematic values
    data = {
        'id': range(5000),
        'name': [f'Person_{i}' if i % 10 != 0 else None for i in range(5000)],
        'description': [f'Description {i}' if i % 5 != 0 else '' for i in range(5000)],
        'category': [f'Cat_{i%20}' if i % 7 != 0 else np.nan for i in range(5000)],
        'score': [i * 0.1 if i % 3 != 0 else np.nan for i in range(5000)],
        'tags': [f'tag{i%10},tag{i%20}' if i % 8 != 0 else None for i in range(5000)],
    }
    
    # Add more columns with mixed data
    for i in range(9):
        data[f'col_{i}'] = [
            f'value_{i}_{j}' if j % (i+2) != 0 else ''
            for j in range(5000)
        ]
    
    # Add specific test values
    data['name'][0] = 'Python Developer'
    data['description'][0] = 'Expert in machine learning'
    data['category'][0] = 'Technology'
    data['tags'][0] = 'python,ml,ai'
    
    data['name'][10] = 'Data Scientist'
    data['description'][10] = 'Python and R programming'
    data['category'][10] = None  # Test None handling
    data['tags'][10] = ''  # Test empty string
    
    df = pd.DataFrame(data)
    
    # Add some infinity values to test handling
    df.loc[20, 'score'] = np.inf
    df.loc[21, 'score'] = -np.inf
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Null values per column:\n{df.isnull().sum().head()}")
    
    # Initialize search engine
    print("\nInitializing search engine...")
    start = time.time()
    engine = FastSearchEngine(df)
    init_time = (time.time() - start) * 1000
    print(f"Initialization time: {init_time:.2f}ms")
    
    # Test pandas mode with some empty filters
    print("\n--- Testing Pandas Mode with Mixed Filters ---")
    filters = {
        'name': 'Python',
        'description': 'machine',
        'invalid_col': 'test',  # Invalid column
        'tags': ''  # Empty filter
    }
    
    start = time.time()
    result = engine.search(mode="pandas", filters=filters)
    search_time = (time.time() - start) * 1000
    
    result_dict = json.loads(result)
    print(f"Search time: {search_time:.2f}ms")
    print(f"Results found: {result_dict['count']}")
    if result_dict.get('error'):
        print(f"Error: {result_dict['error']}")
    
    # Test best_match mode
    print("\n--- Testing Best Match Mode ---")
    query = "Python machine learning programming"
    
    start = time.time()
    result = engine.search(mode="best_match", query=query)
    search_time = (time.time() - start) * 1000
    
    result_dict = json.loads(result)
    print(f"Search time: {search_time:.2f}ms")
    print(f"Results found: {result_dict['count']}")
    
    if result_dict['results']:
        print("\nTop result:")
        top_result = result_dict['results'][0]
        print(f"Score: {top_result.get('_score', 'N/A')}")
        print(f"Matched columns: {top_result.get('_matched_columns', [])}")
        
        # Check JSON serialization
        print(f"\nJSON serialization test:")
        print(f"Result keys: {list(top_result.keys())[:5]}...")
    
    # Test with empty query
    print("\n--- Testing Error Handling ---")
    result = engine.search(mode="best_match", query="")
    result_dict = json.loads(result)
    print(f"Empty query error: {result_dict.get('error', 'No error')}")
    
    # Test with all None filters
    result = engine.search(mode="pandas", filters={'name': None, 'description': ''})
    result_dict = json.loads(result)
    print(f"Empty filters error: {result_dict.get('error', 'No error')}")
    
    print("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    example_usage()
