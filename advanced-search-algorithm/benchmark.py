"""
Benchmark script for the advanced search engine.
Tests performance across different search modes and data sizes.
"""

import os
import time
import random
import statistics
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from search import SearchEngine

# Load environment variables
load_dotenv()

class SearchBenchmark:
    def __init__(self, index_path: str):
        """Initialize benchmark with search engine."""
        self.engine = SearchEngine(index_path)

        # Load sample data for generating queries
        self.data_path = os.getenv("DATA_PATH", "mock_data_small.parquet")
        self.sample_df = pd.read_parquet(self.data_path).head(1000)

    def generate_dataframe_queries(self, num_queries: int) -> List[Dict]:
        """Generate random dataframe search queries."""
        queries = []

        # Common column combinations for filtering
        column_combos = [
            ['status'],
            ['type'],
            ['status', 'type'],
            ['is_active', 'region'],
            ['product', 'category'],
            ['status', 'type', 'priority'],
            ['department', 'is_active'],
            ['region', 'warehouse'],
        ]

        for _ in range(num_queries):
            # Pick random column combination
            columns = random.choice(column_combos)

            # Build query
            query = {}
            for col in columns:
                if col in self.sample_df.columns:
                    # Pick random value from that column
                    non_null_values = self.sample_df[col].dropna().unique()
                    if len(non_null_values) > 0:
                        query[col] = random.choice(non_null_values)

            if query:
                queries.append(query)

        return queries

    def generate_keyword_queries(self, num_queries: int) -> List[List[str]]:
        """Generate random keyword search queries."""
        # Common search terms
        search_terms = [
            ['data', 'search', 'engine'],
            ['machine', 'learning'],
            ['algorithm', 'performance'],
            ['database', 'index'],
            ['query', 'result'],
            ['system', 'processing'],
            ['vector', 'embedding'],
            ['artificial', 'intelligence'],
            ['fast', 'efficient'],
            ['document', 'retrieval']
        ]

        queries = []
        for _ in range(num_queries):
            # Pick 1-3 random keyword sets and combine
            num_sets = random.randint(1, 3)
            selected_terms = []
            for _ in range(num_sets):
                selected_terms.extend(random.choice(search_terms))

            # Randomly select 1-5 keywords
            num_keywords = random.randint(1, min(5, len(selected_terms)))
            keywords = random.sample(selected_terms, num_keywords)
            queries.append(keywords)

        return queries

    def generate_vector_queries(self, num_queries: int) -> List[np.ndarray]:
        """Generate random vector queries."""
        vectors = []
        for _ in range(num_queries):
            # Generate random vector
            vector = np.random.randn(self.engine.vector_dimension).astype(np.float32)
            # Normalize
            vector = vector / np.linalg.norm(vector)
            vectors.append(vector)
        return vectors

    def generate_hybrid_queries(self, num_queries: int) -> List[Dict]:
        """Generate hybrid search queries."""
        queries = []

        df_queries = self.generate_dataframe_queries(num_queries)
        kw_queries = self.generate_keyword_queries(num_queries)
        vec_queries = self.generate_vector_queries(num_queries)

        for i in range(num_queries):
            # Randomly combine different search modes
            query = {}

            # 30% chance to include dataframe search
            if random.random() < 0.3 and i < len(df_queries):
                query['column_value_pairs'] = df_queries[i]

            # 50% chance to include keyword search
            if random.random() < 0.5 and i < len(kw_queries):
                query['keywords'] = kw_queries[i]

            # 40% chance to include vector search
            if random.random() < 0.4 and i < len(vec_queries):
                query['query_vector'] = vec_queries[i]

            # Ensure at least one search mode
            if not query:
                mode = random.choice(['df', 'kw', 'vec'])
                if mode == 'df' and i < len(df_queries):
                    query['column_value_pairs'] = df_queries[i]
                elif mode == 'kw' and i < len(kw_queries):
                    query['keywords'] = kw_queries[i]
                elif i < len(vec_queries):
                    query['query_vector'] = vec_queries[i]

            queries.append(query)

        return queries

    def benchmark_search_mode(
        self,
        search_mode: str,
        queries: List[Any],
        warmup_runs: int = 5
    ) -> Dict[str, Any]:
        """
        Benchmark a specific search mode.

        Args:
            search_mode: Search mode to benchmark
            queries: List of queries for the mode
            warmup_runs: Number of warmup runs

        Returns:
            Benchmark statistics
        """
        print(f"\nBenchmarking {search_mode} search...")

        # Warmup runs
        print(f"Warming up with {warmup_runs} queries...")
        for i in range(min(warmup_runs, len(queries))):
            query = queries[i]
            if search_mode == 'dataframe':
                self.engine.search('dataframe', column_value_pairs=query)
            elif search_mode == 'keyword':
                self.engine.search('keyword', keywords=query)
            elif search_mode == 'vector':
                self.engine.search('vector', query_vector=query)
            elif search_mode == 'hybrid':
                self.engine.search('hybrid', **query)

        # Actual benchmark
        response_times = []
        result_counts = []
        success_count = 0
        error_count = 0

        print(f"Running {len(queries)} queries...")
        for query in tqdm(queries, desc=f"{search_mode} queries"):
            try:
                # Execute search
                if search_mode == 'dataframe':
                    result = self.engine.search('dataframe', column_value_pairs=query)
                elif search_mode == 'keyword':
                    result = self.engine.search('keyword', keywords=query, top_k=5)
                elif search_mode == 'vector':
                    result = self.engine.search('vector', query_vector=query, top_k=5)
                elif search_mode == 'hybrid':
                    result = self.engine.search('hybrid', top_k=5, **query)

                # Record metrics
                response_times.append(result['time_taken'] * 1000)  # Convert to ms
                result_counts.append(result['count'])

                if result['status'] == 'success':
                    success_count += 1
                else:
                    error_count += 1

            except Exception as e:
                print(f"Error in query: {e}")
                error_count += 1
                response_times.append(1000)  # Default 1 second for errors

        # Calculate statistics
        stats = {
            'mode': search_mode,
            'total_queries': len(queries),
            'success_count': success_count,
            'error_count': error_count,
            'success_rate': (success_count / len(queries)) * 100,
            'avg_response_time_ms': statistics.mean(response_times),
            'median_response_time_ms': statistics.median(response_times),
            'min_response_time_ms': min(response_times),
            'max_response_time_ms': max(response_times),
            'p95_response_time_ms': np.percentile(response_times, 95),
            'p99_response_time_ms': np.percentile(response_times, 99),
            'avg_result_count': statistics.mean(result_counts) if result_counts else 0,
            'queries_under_200ms': sum(1 for t in response_times if t < 200),
            'percentage_under_200ms': (sum(1 for t in response_times if t < 200) / len(response_times)) * 100
        }

        return stats

    def run_comprehensive_benchmark(self, num_queries_per_mode: int = 100):
        """
        Run comprehensive benchmark across all search modes.

        Args:
            num_queries_per_mode: Number of queries to run per mode
        """
        print("\n" + "="*60)
        print("ADVANCED SEARCH ENGINE BENCHMARK")
        print("="*60)

        print(f"\nEngine Configuration:")
        print(f"  Total Documents: {self.engine.total_rows:,}")
        print(f"  Total Columns: {len(self.engine.columns)}")
        print(f"  Vector Dimension: {self.engine.vector_dimension}")
        print(f"  Embedding Columns: {self.engine.embedding_columns}")

        all_stats = []

        # 1. Benchmark Dataframe Search
        df_queries = self.generate_dataframe_queries(num_queries_per_mode)
        df_stats = self.benchmark_search_mode('dataframe', df_queries)
        all_stats.append(df_stats)

        # 2. Benchmark Keyword Search
        kw_queries = self.generate_keyword_queries(num_queries_per_mode)
        kw_stats = self.benchmark_search_mode('keyword', kw_queries)
        all_stats.append(kw_stats)

        # 3. Benchmark Vector Search
        vec_queries = self.generate_vector_queries(num_queries_per_mode)
        vec_stats = self.benchmark_search_mode('vector', vec_queries)
        all_stats.append(vec_stats)

        # 4. Benchmark Hybrid Search
        hybrid_queries = self.generate_hybrid_queries(num_queries_per_mode)
        hybrid_stats = self.benchmark_search_mode('hybrid', hybrid_queries)
        all_stats.append(hybrid_stats)

        # Print summary
        self.print_benchmark_summary(all_stats)

        return all_stats

    def print_benchmark_summary(self, stats: List[Dict[str, Any]]):
        """Print formatted benchmark summary."""
        print("\n" + "="*60)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*60)

        # Create summary table
        headers = ['Mode', 'Queries', 'Success%', 'Avg(ms)', 'P95(ms)', 'P99(ms)', '<200ms%']

        print("\n" + " | ".join(f"{h:^10}" for h in headers))
        print("-" * (len(headers) * 12))

        for stat in stats:
            row = [
                stat['mode'].capitalize(),
                str(stat['total_queries']),
                f"{stat['success_rate']:.1f}%",
                f"{stat['avg_response_time_ms']:.1f}",
                f"{stat['p95_response_time_ms']:.1f}",
                f"{stat['p99_response_time_ms']:.1f}",
                f"{stat['percentage_under_200ms']:.1f}%"
            ]
            print(" | ".join(f"{r:^10}" for r in row))

        # Performance requirements check
        print("\n" + "="*60)
        print("PERFORMANCE REQUIREMENTS CHECK")
        print("="*60)

        requirement_met = True
        for stat in stats:
            if stat['percentage_under_200ms'] >= 95:
                status = "✓ PASS"
            else:
                status = "✗ FAIL"
                requirement_met = False

            print(f"{stat['mode'].capitalize():15} {status:10} "
                  f"({stat['percentage_under_200ms']:.1f}% queries < 200ms)")

        print("\n" + "="*60)
        if requirement_met:
            print("✓ ALL PERFORMANCE REQUIREMENTS MET")
        else:
            print("✗ SOME PERFORMANCE REQUIREMENTS NOT MET")
        print("="*60)

def main():
    """Main benchmark execution."""
    # Load configuration
    load_dotenv()

    index_path = os.getenv("INDEX_PATH", "./indices")

    # Check if indices exist
    if not os.path.exists(index_path):
        print("Error: Index directory not found. Please run ingestion first.")
        return

    # Initialize benchmark
    benchmark = SearchBenchmark(index_path)

    # Run comprehensive benchmark
    num_queries = int(os.getenv("BENCHMARK_QUERIES", "100"))
    stats = benchmark.run_comprehensive_benchmark(num_queries)

    # Save results to file
    results_file = "benchmark_results.json"
    import json
    with open(results_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nBenchmark results saved to {results_file}")

if __name__ == "__main__":
    main()