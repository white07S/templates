import faiss
import numpy as np
import pandas as pd
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import json


@dataclass
class FeatureIndex:
    """Container for a single feature's FAISS index and metadata"""
    name: str
    index: faiss.IndexFlatL2
    hash_mapping: Dict[int, str]  # Maps FAISS index position to original hash
    is_list_feature: bool = False
    list_boundaries: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # For list features: hash -> (start_idx, end_idx)


class MultiFeatureSearchEngine:
    """
    FAISS-based search engine for multi-feature embeddings with intelligent ranking.
    Supports both single embeddings and lists of embeddings per feature.
    """
    
    def __init__(self, 
                 embedding_dim: int = 4096,
                 cache_dir: str = "./cache",
                 k_rrf: int = 60):  # RRF constant
        """
        Initialize the search engine.
        
        Args:
            embedding_dim: Dimension of embeddings (default 4096)
            cache_dir: Directory to cache embeddings
            k_rrf: Constant for Reciprocal Rank Fusion (default 60, recommended value)
        """
        self.embedding_dim = embedding_dim
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.k_rrf = k_rrf
        
        # Feature indices
        self.feature_indices: Dict[str, FeatureIndex] = {}
        
        # Cache for embeddings
        self.embeddings_cache_file = self.cache_dir / "embeddings_cache.pkl"
        self.index_cache_file = self.cache_dir / "indices_cache.pkl"
        
        # Load cached data if exists
        self._load_cache()
    
    def _load_cache(self):
        """Load cached embeddings and indices if they exist"""
        if self.index_cache_file.exists():
            try:
                with open(self.index_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.feature_indices = cache_data['feature_indices']
                    print(f"Loaded cached indices for features: {list(self.feature_indices.keys())}")
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.feature_indices = {}
    
    def _save_cache(self):
        """Save indices to cache"""
        try:
            cache_data = {
                'feature_indices': self.feature_indices
            }
            with open(self.index_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print("Saved indices to cache")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def _get_embedding_cache_key(self, text: str, feature_name: str) -> str:
        """Generate a unique cache key for text+feature combination"""
        content = f"{feature_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_embeddings_cache(self) -> Dict[str, np.ndarray]:
        """Load embeddings cache from disk"""
        if self.embeddings_cache_file.exists():
            try:
                with open(self.embeddings_cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def _save_embeddings_cache(self, cache: Dict[str, np.ndarray]):
        """Save embeddings cache to disk"""
        with open(self.embeddings_cache_file, 'wb') as f:
            pickle.dump(cache, f)
    
    def get_embedding(self, text: Union[str, List[str]], feature_name: str) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Get embedding for text(s). Uses cache if available.
        
        This is a placeholder - replace with actual embedding function.
        
        Args:
            text: Single string or list of strings
            feature_name: Name of the feature (for caching)
            
        Returns:
            Single embedding or list of embeddings
        """
        embeddings_cache = self._load_embeddings_cache()
        
        if isinstance(text, str):
            cache_key = self._get_embedding_cache_key(text, feature_name)
            if cache_key in embeddings_cache:
                return embeddings_cache[cache_key]
            
            # Placeholder: Replace this with actual embedding generation
            # embedding = your_embedding_function(text)
            embedding = np.random.randn(self.embedding_dim).astype('float32')  # Placeholder
            
            embeddings_cache[cache_key] = embedding
            self._save_embeddings_cache(embeddings_cache)
            return embedding
        
        else:  # List of strings
            embeddings = []
            cache_updated = False
            
            for t in text:
                cache_key = self._get_embedding_cache_key(t, feature_name)
                if cache_key in embeddings_cache:
                    embeddings.append(embeddings_cache[cache_key])
                else:
                    # Placeholder: Replace this with actual embedding generation
                    # embedding = your_embedding_function(t)
                    embedding = np.random.randn(self.embedding_dim).astype('float32')  # Placeholder
                    
                    embeddings_cache[cache_key] = embedding
                    embeddings.append(embedding)
                    cache_updated = True
            
            if cache_updated:
                self._save_embeddings_cache(embeddings_cache)
            
            return embeddings
    
    def ingest_and_index(self, df: pd.DataFrame, force_reindex: bool = False):
        """
        Ingest data from DataFrame and create FAISS indices.
        
        Args:
            df: DataFrame with columns: Hash, summary, control_deficiency, 
                problem_statements (List[str]), issue_failing (List[str])
            force_reindex: Force re-indexing even if cache exists
        """
        if self.feature_indices and not force_reindex:
            print("Indices already exist. Use force_reindex=True to rebuild.")
            return
        
        print("Starting ingestion and indexing...")
        
        # Process single embedding features
        single_features = ['summary', 'control_deficiency']
        for feature_name in single_features:
            print(f"Processing feature: {feature_name}")
            
            index = faiss.IndexFlatL2(self.embedding_dim)
            hash_mapping = {}
            
            embeddings_list = []
            for idx, row in df.iterrows():
                text = row[feature_name]
                if pd.notna(text) and text:
                    embedding = self.get_embedding(text, feature_name)
                    embeddings_list.append(embedding)
                    hash_mapping[len(embeddings_list) - 1] = row['Hash']
            
            if embeddings_list:
                embeddings_array = np.vstack(embeddings_list)
                index.add(embeddings_array)
            
            self.feature_indices[feature_name] = FeatureIndex(
                name=feature_name,
                index=index,
                hash_mapping=hash_mapping,
                is_list_feature=False
            )
        
        # Process list embedding features
        list_features = ['problem_statements', 'issue_failing']
        for feature_name in list_features:
            print(f"Processing list feature: {feature_name}")
            
            index = faiss.IndexFlatL2(self.embedding_dim)
            hash_mapping = {}
            list_boundaries = {}
            
            embeddings_list = []
            for idx, row in df.iterrows():
                text_list = row[feature_name]
                if pd.notna(text_list) and text_list:
                    start_idx = len(embeddings_list)
                    
                    embeddings = self.get_embedding(text_list, feature_name)
                    for emb in embeddings:
                        embeddings_list.append(emb)
                        hash_mapping[len(embeddings_list) - 1] = row['Hash']
                    
                    end_idx = len(embeddings_list)
                    list_boundaries[row['Hash']] = (start_idx, end_idx)
            
            if embeddings_list:
                embeddings_array = np.vstack(embeddings_list)
                index.add(embeddings_array)
            
            self.feature_indices[feature_name] = FeatureIndex(
                name=feature_name,
                index=index,
                hash_mapping=hash_mapping,
                is_list_feature=True,
                list_boundaries=list_boundaries
            )
        
        self._save_cache()
        print(f"Indexing complete. Created indices for: {list(self.feature_indices.keys())}")
    
    def _search_single_feature(self, 
                              query_vector: np.ndarray, 
                              feature_index: FeatureIndex, 
                              k: int = 5) -> List[Tuple[str, float]]:
        """
        Search a single feature index with a query vector.
        
        Returns:
            List of (hash, distance) tuples
        """
        if feature_index.index.ntotal == 0:
            return []
        
        # Ensure query vector is 2D
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Search
        distances, indices = feature_index.index.search(query_vector, min(k, feature_index.index.ntotal))
        
        results = []
        seen_hashes = set()
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx in feature_index.hash_mapping:
                hash_val = feature_index.hash_mapping[idx]
                if hash_val not in seen_hashes:
                    results.append((hash_val, float(dist)))
                    seen_hashes.add(hash_val)
        
        return results
    
    def _reciprocal_rank_fusion(self, 
                                results_dict: Dict[str, List[Tuple[str, float]]]) -> List[str]:
        """
        Apply Reciprocal Rank Fusion to merge results from multiple features.
        
        Args:
            results_dict: Dictionary mapping feature_name -> [(hash, distance), ...]
            
        Returns:
            Top 5 hashes ranked by RRF score
        """
        rrf_scores = defaultdict(float)
        
        for feature_name, results in results_dict.items():
            for rank, (hash_val, distance) in enumerate(results):
                # RRF formula: 1 / (rank + k)
                rrf_score = 1.0 / (rank + 1 + self.k_rrf)
                rrf_scores[hash_val] += rrf_score
        
        # Sort by RRF score (descending)
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 5 hashes
        return [hash_val for hash_val, _ in sorted_results[:5]]
    
    def search(self, 
              query: Union[np.ndarray, List[np.ndarray]], 
              features_to_search: Optional[List[str]] = None,
              k_per_feature: int = 5) -> List[str]:
        """
        Search across multiple features with intelligent ranking.
        
        Args:
            query: Single vector or list of vectors (4096-dim)
            features_to_search: List of features to search (None = all features)
            k_per_feature: Number of top results per feature (default 5)
            
        Returns:
            Top 5 hashes based on intelligent ranking
        """
        if not self.feature_indices:
            raise ValueError("No indices available. Please run ingest_and_index first.")
        
        if features_to_search is None:
            features_to_search = list(self.feature_indices.keys())
        
        # Handle single vector query
        if isinstance(query, np.ndarray) and query.ndim == 1:
            query = [query]
        elif isinstance(query, np.ndarray) and query.ndim == 2:
            query = [query[i] for i in range(query.shape[0])]
        
        # Collect results from all features for all query vectors
        all_results = defaultdict(lambda: defaultdict(list))  # query_idx -> feature -> results
        
        for query_idx, query_vector in enumerate(query):
            for feature_name in features_to_search:
                if feature_name not in self.feature_indices:
                    continue
                
                feature_index = self.feature_indices[feature_name]
                results = self._search_single_feature(query_vector, feature_index, k_per_feature)
                all_results[query_idx][feature_name] = results
        
        # Merge results across query vectors
        if len(query) == 1:
            # Single query vector - use standard RRF
            return self._reciprocal_rank_fusion(all_results[0])
        else:
            # Multiple query vectors - aggregate RRF scores
            aggregated_scores = defaultdict(float)
            
            for query_idx in all_results:
                query_results = all_results[query_idx]
                rrf_results = self._reciprocal_rank_fusion(query_results)
                
                # Weight by position in RRF results
                for rank, hash_val in enumerate(rrf_results):
                    aggregated_scores[hash_val] += 1.0 / (rank + 1)
            
            # Sort by aggregated score
            sorted_results = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)
            return [hash_val for hash_val, _ in sorted_results[:5]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the indexed data"""
        stats = {}
        for feature_name, feature_index in self.feature_indices.items():
            stats[feature_name] = {
                'total_vectors': feature_index.index.ntotal,
                'unique_hashes': len(set(feature_index.hash_mapping.values())),
                'is_list_feature': feature_index.is_list_feature
            }



import numpy as np
import pandas as pd
from multi_feature_search import MultiFeatureSearchEngine
from typing import Union, List
import time


def get_embedding_function(text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Your actual embedding function goes here.
    Replace this with your actual implementation.
    
    Args:
        text: Single string or list of strings
        
    Returns:
        4096-dimensional embedding(s)
    """
    # PLACEHOLDER: Replace with your actual embedding function
    # Example: return model.encode(text)
    
    if isinstance(text, str):
        # Single text - return single embedding
        return np.random.randn(4096).astype('float32')
    else:
        # List of texts - return list of embeddings
        return [np.random.randn(4096).astype('float32') for _ in text]


def create_sample_data(n_rows: int = 100) -> pd.DataFrame:
    """Create sample data for testing"""
    data = []
    for i in range(n_rows):
        data.append({
            'Hash': f'hash_{i:04d}',
            'summary': f'This is a summary for item {i}',
            'control_deficiency': f'Control deficiency description for item {i}',
            'problem_statements': [f'Problem {j} for item {i}' for j in range(np.random.randint(1, 4))],
            'issue_failing': [f'Issue {j} for item {i}' for j in range(np.random.randint(1, 3))]
        })
    return pd.DataFrame(data)


def main():
    """Main demonstration of the multi-feature search engine"""
    
    print("=" * 80)
    print("FAISS Multi-Feature Search Engine")
    print("=" * 80)
    
    # Initialize the search engine
    engine = MultiFeatureSearchEngine(
        embedding_dim=4096,
        cache_dir="./cache",
        k_rrf=60  # RRF constant
    )
    
    # IMPORTANT: Replace the placeholder embedding function in multi_feature_search.py
    # with your actual embedding function by modifying the get_embedding method
    
    # Create or load your data
    print("\n1. Loading data...")
    # Option 1: Load from parquet
    # df = pd.read_parquet('your_data.parquet')
    
    # Option 2: Create sample data for testing
    df = create_sample_data(100)
    print(f"Loaded {len(df)} rows")
    
    # Ingest and index the data
    print("\n2. Ingesting and indexing data...")
    start_time = time.time()
    engine.ingest_and_index(df, force_reindex=False)  # Set to True to force re-indexing
    print(f"Indexing completed in {time.time() - start_time:.2f} seconds")
    
    # Show statistics
    print("\n3. Index Statistics:")
    stats = engine.get_statistics()
    for feature, feature_stats in stats.items():
        print(f"   {feature}:")
        print(f"      Total vectors: {feature_stats['total_vectors']}")
        print(f"      Unique hashes: {feature_stats['unique_hashes']}")
        print(f"      Is list feature: {feature_stats['is_list_feature']}")
    
    # Example 1: Single vector query
    print("\n4. Testing single vector query...")
    query_vector = get_embedding_function("Sample query text about control issues")
    
    start_time = time.time()
    results = engine.search(query_vector, k_per_feature=5)
    search_time = time.time() - start_time
    
    print(f"   Search completed in {search_time*1000:.2f} ms")
    print(f"   Top 5 results: {results}")
    
    # Example 2: Multiple vector query
    print("\n5. Testing multiple vector query...")
    query_texts = [
        "Control deficiency in authentication",
        "Problem with data validation",
        "Issue with access control"
    ]
    query_vectors = get_embedding_function(query_texts)
    
    start_time = time.time()
    results = engine.search(query_vectors, k_per_feature=5)
    search_time = time.time() - start_time
    
    print(f"   Search completed in {search_time*1000:.2f} ms")
    print(f"   Top 5 results (merged from multiple queries): {results}")
    
    # Example 3: Search specific features only
    print("\n6. Testing search on specific features...")
    query_vector = get_embedding_function("Security vulnerability")
    
    start_time = time.time()
    results = engine.search(
        query_vector, 
        features_to_search=['summary', 'control_deficiency'],
        k_per_feature=3
    )
    search_time = time.time() - start_time
    
    print(f"   Search completed in {search_time*1000:.2f} ms")
    print(f"   Top 5 results (from selected features): {results}")
    
    # Show how caching works
    print("\n7. Demonstrating caching...")
    print("   Running the same query again (should use cache)...")
    
    start_time = time.time()
    results = engine.search(query_vector, k_per_feature=5)
    search_time = time.time() - start_time
    
    print(f"   Search completed in {search_time*1000:.2f} ms (faster due to caching)")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
        return stats
