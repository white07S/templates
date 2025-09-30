"""Benchmark script for the search engine."""

import time
import numpy as np
import statistics
from typing import List, Dict, Any
from tqdm import tqdm
import argparse
import json

from src.core.models import SearchMode, SearchRequest
from src.core.config import SearchConfig
from src.search.search_engine import SearchEngine


class Benchmark:
    """Benchmark suite for search engine performance testing."""

    def __init__(self, engine: SearchEngine):
        """Initialize benchmark.

        Args:
            engine: Search engine instance
        """
        self.engine = engine
        self.results = {}

    def generate_test_queries(self, num_queries: int = 100) -> Dict[str, List[Dict]]:
        """Generate test queries for each search mode.

        Args:
            num_queries: Number of queries per mode

        Returns:
            Dictionary of mode -> list of query parameters
        """
        queries = {
            "dataframe": [],
            "keyword": [],
            "vector": [],
            "hybrid": []
        }

        # Common values for testing
        categories = ['Technology', 'Science', 'Business', 'Health', 'Education']
        statuses = ['Active', 'Pending', 'Completed', 'Cancelled', 'Draft']
        priorities = ['Low', 'Medium', 'High', 'Critical']
        countries = ['USA', 'Canada', 'UK', 'Germany', 'France']

        # Generate dataframe queries
        for _ in range(num_queries):
            query = {
                "column_value_pairs": {
                    np.random.choice(['category', 'status', 'priority', 'country']):
                    np.random.choice([categories, statuses, priorities, countries][0])
                }
            }
            queries["dataframe"].append(query)

        # Generate keyword queries
        keywords_pool = [
            ["technology", "innovation"],
            ["business", "strategy"],
            ["health", "medical"],
            ["education", "learning"],
            ["science", "research"],
            ["data", "analysis"],
            ["market", "growth"],
            ["product", "development"],
            ["customer", "service"],
            ["digital", "transformation"]
        ]

        for _ in range(num_queries):
            query = {
                "keywords": np.random.choice(keywords_pool)
            }
            queries["keyword"].append(query)

        # Generate vector queries
        for _ in range(num_queries):
            query = {
                "vector": np.random.randn(4096).tolist()
            }
            queries["vector"].append(query)

        # Generate hybrid queries
        for _ in range(num_queries):
            query = {}

            # Randomly include different search parameters
            if np.random.random() > 0.3:
                query["column_value_pairs"] = {
                    "category": np.random.choice(categories)
                }
            if np.random.random() > 0.3:
                query["keywords"] = np.random.choice(keywords_pool)
            if np.random.random() > 0.3:
                query["vector"] = np.random.randn(4096).tolist()

            # Ensure at least one parameter
            if not query:
                query["keywords"] = ["search", "query"]

            queries["hybrid"].append(query)

        return queries

    def benchmark_mode(self, mode: SearchMode, queries: List[Dict], top_k: int = 5) -> Dict[str, Any]:
        """Benchmark a specific search mode.

        Args:
            mode: Search mode to benchmark
            queries: List of query parameters
            top_k: Number of results to retrieve

        Returns:
            Benchmark statistics
        """
        print(f"\nBenchmarking {mode} mode with {len(queries)} queries...")

        latencies = []
        successful = 0
        failed = 0
        result_counts = []

        for query_params in tqdm(queries, desc=f"Running {mode} queries"):
            # Create search request
            request = SearchRequest(mode=mode, top_k=top_k, **query_params)

            # Perform search and measure time
            start_time = time.perf_counter()
            result = self.engine.search(request)
            latency = (time.perf_counter() - start_time) * 1000  # Convert to ms

            latencies.append(latency)

            if result.status == "success":
                successful += 1
                result_counts.append(result.count)
            else:
                failed += 1
                result_counts.append(0)

        # Calculate statistics
        stats = {
            "mode": mode,
            "num_queries": len(queries),
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / len(queries)) * 100,
            "latency_mean": statistics.mean(latencies),
            "latency_median": statistics.median(latencies),
            "latency_min": min(latencies),
            "latency_max": max(latencies),
            "latency_p95": np.percentile(latencies, 95),
            "latency_p99": np.percentile(latencies, 99),
            "avg_results": statistics.mean(result_counts) if result_counts else 0,
            "under_200ms": sum(1 for l in latencies if l < 200),
            "under_200ms_pct": (sum(1 for l in latencies if l < 200) / len(latencies)) * 100
        }

        return stats

    def run_benchmark(self, num_queries: int = 100, top_k: int = 5):
        """Run full benchmark suite.

        Args:
            num_queries: Number of queries per mode
            top_k: Number of results to retrieve
        """
        print("=" * 80)
        print("SEARCH ENGINE PERFORMANCE BENCHMARK")
        print("=" * 80)

        # Generate test queries
        print(f"\nGenerating {num_queries} test queries per mode...")
        queries = self.generate_test_queries(num_queries)

        # Benchmark each mode
        modes = [SearchMode.DATAFRAME, SearchMode.KEYWORD, SearchMode.VECTOR, SearchMode.HYBRID]

        for mode in modes:
            stats = self.benchmark_mode(mode, queries[mode], top_k)
            self.results[mode] = stats

        # Print results
        self.print_results()

    def print_results(self):
        """Print benchmark results in a formatted table."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)

        # Print header
        print(f"\n{'Mode':<12} {'Queries':<8} {'Success':<8} {'Mean(ms)':<10} "
              f"{'P95(ms)':<10} {'P99(ms)':<10} {'<200ms':<10}")
        print("-" * 80)

        # Print results for each mode
        for mode, stats in self.results.items():
            print(f"{mode:<12} {stats['num_queries']:<8} "
                  f"{stats['success_rate']:.1f}%{'':<3} "
                  f"{stats['latency_mean']:.1f}{'':<5} "
                  f"{stats['latency_p95']:.1f}{'':<5} "
                  f"{stats['latency_p99']:.1f}{'':<5} "
                  f"{stats['under_200ms_pct']:.1f}%")

        # Print detailed statistics
        print("\n" + "=" * 80)
        print("DETAILED STATISTICS")
        print("=" * 80)

        for mode, stats in self.results.items():
            print(f"\n{mode.upper()} MODE:")
            print(f"  - Total queries: {stats['num_queries']}")
            print(f"  - Successful: {stats['successful']} ({stats['success_rate']:.1f}%)")
            print(f"  - Failed: {stats['failed']}")
            print(f"  - Average results per query: {stats['avg_results']:.1f}")
            print(f"  - Latency statistics:")
            print(f"    * Mean: {stats['latency_mean']:.2f} ms")
            print(f"    * Median: {stats['latency_median']:.2f} ms")
            print(f"    * Min: {stats['latency_min']:.2f} ms")
            print(f"    * Max: {stats['latency_max']:.2f} ms")
            print(f"    * P95: {stats['latency_p95']:.2f} ms")
            print(f"    * P99: {stats['latency_p99']:.2f} ms")
            print(f"  - Queries under 200ms: {stats['under_200ms']} ({stats['under_200ms_pct']:.1f}%)")

        # Overall assessment
        print("\n" + "=" * 80)
        print("PERFORMANCE ASSESSMENT")
        print("=" * 80)

        total_under_200ms = sum(s['under_200ms'] for s in self.results.values())
        total_queries = sum(s['num_queries'] for s in self.results.values())
        overall_pct = (total_under_200ms / total_queries) * 100

        print(f"\nOverall Performance:")
        print(f"  - Total queries: {total_queries}")
        print(f"  - Queries under 200ms: {total_under_200ms} ({overall_pct:.1f}%)")

        if overall_pct >= 95:
            print("  ✓ MEETS REQUIREMENT: 95% of queries complete under 200ms")
        else:
            print(f"  ✗ BELOW REQUIREMENT: Only {overall_pct:.1f}% under 200ms (need 95%)")

    def save_results(self, filename: str):
        """Save benchmark results to JSON file.

        Args:
            filename: Output filename
        """
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filename}")


def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(description="Benchmark search engine performance")

    parser.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Number of queries per search mode"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve"
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="data/index",
        help="Path to index files"
    )
    parser.add_argument(
        "--save-results",
        type=str,
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    # Create configuration
    config = SearchConfig(
        index_path=args.index_path,
        top_k=args.top_k
    )

    # Initialize search engine
    print("Initializing search engine...")
    engine = SearchEngine(config)

    # Run benchmark
    benchmark = Benchmark(engine)
    benchmark.run_benchmark(num_queries=args.num_queries, top_k=args.top_k)

    # Save results if requested
    if args.save_results:
        benchmark.save_results(args.save_results)


if __name__ == "__main__":
    main()