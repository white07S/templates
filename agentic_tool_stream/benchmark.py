#!/usr/bin/env python3
"""
Performance Benchmark Script for Search Engine
Tests all search modes and measures performance metrics
"""

import time
import numpy as np
import json
import random
from search_engine import SearchEngine
from embeddings_client import EmbeddingsClient
import statistics

class SearchBenchmark:
    def __init__(self, index_path="indices", num_queries=100):
        self.index_path = index_path
        self.num_queries = num_queries
        self.engine = SearchEngine(index_path=index_path)
        self.embeddings_client = EmbeddingsClient(
            dimension=4096,
            cache_dir=f"{index_path}/embeddings_cache"
        )

        # Sample data for testing
        self.sample_categories = [f"category_{i}" for i in range(50)]
        self.sample_keywords = [
            "data", "search", "engine", "performance", "optimization",
            "query", "algorithm", "index", "database", "system"
        ]

    def benchmark_dataframe(self):
        """Benchmark dataframe search mode"""
        print("\n" + "=" * 60)
        print("DATAFRAME MODE BENCHMARK")
        print("=" * 60)

        times = []
        success_count = 0

        for i in range(self.num_queries):
            # Generate random column-value pairs
            column_value_pairs = {
                f"category_{random.randint(1, 10)}": random.choice(self.sample_categories),
                f"integer_{random.randint(1, 5)}": random.randint(0, 1000)
            }

            start = time.perf_counter()
            result = self.engine.search_dataframe(column_value_pairs)
            elapsed = time.perf_counter() - start

            times.append(elapsed)
            if result['status'] == 'success':
                success_count += 1

            if (i + 1) % 20 == 0:
                print(f"Progress: {i + 1}/{self.num_queries} queries completed")

        self._print_stats("Dataframe", times, success_count)
        return times

    def benchmark_keyword(self):
        """Benchmark keyword search mode"""
        print("\n" + "=" * 60)
        print("KEYWORD MODE BENCHMARK")
        print("=" * 60)

        times = []
        success_count = 0

        for i in range(self.num_queries):
            # Generate random keyword combinations
            num_keywords = random.randint(1, 3)
            keywords = random.sample(self.sample_keywords, num_keywords)

            start = time.perf_counter()
            result = self.engine.search_keyword(keywords)
            elapsed = time.perf_counter() - start

            times.append(elapsed)
            if result['status'] == 'success':
                success_count += 1

            if (i + 1) % 20 == 0:
                print(f"Progress: {i + 1}/{self.num_queries} queries completed")

        self._print_stats("Keyword", times, success_count)
        return times

    def benchmark_vector(self):
        """Benchmark vector search mode"""
        print("\n" + "=" * 60)
        print("VECTOR MODE BENCHMARK")
        print("=" * 60)

        times = []
        success_count = 0

        # Pre-generate random vectors
        base_vector = np.random.randn(4096).astype(np.float32)
        base_vector = base_vector / np.linalg.norm(base_vector)

        for i in range(self.num_queries):
            # Add small noise to base vector
            noise = np.random.randn(4096) * 0.05
            query_vector = base_vector + noise
            query_vector = query_vector / np.linalg.norm(query_vector)

            start = time.perf_counter()
            result = self.engine.search_vector(query_vector)
            elapsed = time.perf_counter() - start

            times.append(elapsed)
            if result['status'] == 'success':
                success_count += 1

            if (i + 1) % 20 == 0:
                print(f"Progress: {i + 1}/{self.num_queries} queries completed")

        self._print_stats("Vector", times, success_count)
        return times

    def benchmark_hybrid(self):
        """Benchmark hybrid search mode"""
        print("\n" + "=" * 60)
        print("HYBRID MODE BENCHMARK")
        print("=" * 60)

        times = []
        success_count = 0

        # Pre-generate base vector
        base_vector = np.random.randn(4096).astype(np.float32)
        base_vector = base_vector / np.linalg.norm(base_vector)

        for i in range(self.num_queries):
            # Mix different search parameters
            params = {}

            # Sometimes include dataframe search
            if random.random() > 0.5:
                params['column_value_pairs'] = {
                    f"category_{random.randint(1, 10)}": random.choice(self.sample_categories)
                }

            # Sometimes include keyword search
            if random.random() > 0.5:
                params['keywords'] = random.sample(self.sample_keywords, random.randint(1, 2))

            # Sometimes include vector search
            if random.random() > 0.5:
                noise = np.random.randn(4096) * 0.05
                query_vector = base_vector + noise
                params['query_vector'] = query_vector / np.linalg.norm(query_vector)

            # Ensure at least one parameter
            if not params:
                params['keywords'] = [random.choice(self.sample_keywords)]

            start = time.perf_counter()
            result = self.engine.search_hybrid(**params)
            elapsed = time.perf_counter() - start

            times.append(elapsed)
            if result['status'] == 'success':
                success_count += 1

            if (i + 1) % 20 == 0:
                print(f"Progress: {i + 1}/{self.num_queries} queries completed")

        self._print_stats("Hybrid", times, success_count)
        return times

    def _print_stats(self, mode, times, success_count):
        """Print statistics for benchmark results"""
        times_ms = [t * 1000 for t in times]  # Convert to milliseconds

        print(f"\nResults for {mode} Mode:")
        print("-" * 40)
        print(f"Total queries: {self.num_queries}")
        print(f"Successful queries: {success_count} ({success_count/self.num_queries*100:.1f}%)")
        print(f"Mean time: {statistics.mean(times_ms):.2f} ms")
        print(f"Median time: {statistics.median(times_ms):.2f} ms")
        print(f"Min time: {min(times_ms):.2f} ms")
        print(f"Max time: {max(times_ms):.2f} ms")

        if len(times_ms) > 1:
            print(f"Std dev: {statistics.stdev(times_ms):.2f} ms")

        # Calculate percentiles
        sorted_times = sorted(times_ms)
        p50 = sorted_times[int(len(sorted_times) * 0.50)]
        p90 = sorted_times[int(len(sorted_times) * 0.90)]
        p95 = sorted_times[int(len(sorted_times) * 0.95)]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]

        print(f"50th percentile: {p50:.2f} ms")
        print(f"90th percentile: {p90:.2f} ms")
        print(f"95th percentile: {p95:.2f} ms")
        print(f"99th percentile: {p99:.2f} ms")

        # Check if meets requirement (<200ms for 95% of queries)
        target_met = p95 < 200
        print(f"\n✓ Target (<200ms for 95% queries): {'PASSED' if target_met else 'FAILED'}")
        print(f"  95th percentile: {p95:.2f} ms {'✓' if p95 < 200 else '✗'}")

    def run_all(self):
        """Run all benchmarks"""
        print("\n" + "=" * 60)
        print("SEARCH ENGINE PERFORMANCE BENCHMARK")
        print("=" * 60)
        print(f"Index path: {self.index_path}")
        print(f"Queries per mode: {self.num_queries}")

        all_results = {}

        # Run each benchmark
        all_results['dataframe'] = self.benchmark_dataframe()
        all_results['keyword'] = self.benchmark_keyword()
        all_results['vector'] = self.benchmark_vector()
        all_results['hybrid'] = self.benchmark_hybrid()

        # Overall summary
        print("\n" + "=" * 60)
        print("OVERALL SUMMARY")
        print("=" * 60)

        for mode, times in all_results.items():
            times_ms = [t * 1000 for t in times]
            p95 = sorted(times_ms)[int(len(times_ms) * 0.95)]
            status = "✓ PASS" if p95 < 200 else "✗ FAIL"
            print(f"{mode.capitalize():12} - 95th percentile: {p95:6.2f} ms {status}")

        # Check if all modes meet the requirement
        all_pass = all(
            sorted([t * 1000 for t in times])[int(len(times) * 0.95)] < 200
            for times in all_results.values()
        )

        print("\n" + "=" * 60)
        if all_pass:
            print("✓ ALL TESTS PASSED - System meets performance requirements!")
        else:
            print("✗ SOME TESTS FAILED - Performance optimization needed")
        print("=" * 60)

        # Save results to file
        results_file = f"benchmark_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'num_queries': self.num_queries,
                'results': {
                    mode: {
                        'times_ms': [t * 1000 for t in times],
                        'mean_ms': statistics.mean([t * 1000 for t in times]),
                        'p95_ms': sorted([t * 1000 for t in times])[int(len(times) * 0.95)]
                    }
                    for mode, times in all_results.items()
                }
            }, f, indent=2)

        print(f"\nResults saved to: {results_file}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Search Engine Performance Benchmark")
    parser.add_argument(
        "--index-path",
        type=str,
        default="indices",
        help="Path to index files"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Number of queries to run per mode"
    )

    args = parser.parse_args()

    try:
        benchmark = SearchBenchmark(
            index_path=args.index_path,
            num_queries=args.num_queries
        )
        benchmark.run_all()
        benchmark.engine.close()
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure indices are created by running engine.py first.")
        return 1

if __name__ == "__main__":
    main()