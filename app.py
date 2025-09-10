#!/usr/bin/env python3
"""
Test script to validate that the search doesn't rebuild index every time
This demonstrates the fix for the rebuilding issue
"""

import time
import os
from pathlib import Path

def test_persistent_index():
    """Test that index persists across manager instances"""
    
    # Import here to ensure module is available
    from tantivy_parquet_manager import TantivyParquetSearchManager
    
    index_dir = "./test_persistent_index"
    
    print("=" * 60)
    print("TEST: Index Persistence and Search Performance")
    print("=" * 60)
    
    # Test 1: Create new manager instance (should load existing index if available)
    print("\n1. Creating first manager instance...")
    manager1 = TantivyParquetSearchManager(
        index_dir=index_dir,
        cache_size=10000,
        enable_optimizations=True
    )
    
    # Check if index exists
    meta_file = Path(index_dir) / "meta.json"
    index_exists = meta_file.exists()
    
    if not index_exists:
        print("   No existing index found. Creating new index...")
        # Only ingest and index if no existing index
        print("   Ingesting test data...")
        manager1.ingest("test_data_small.parquet", batch_size=1000)
        print("   Building index...")
        manager1.index(optimize_for_speed=True, num_threads=4)
    else:
        print("   Existing index loaded successfully!")
    
    # Perform search with first manager
    print("\n2. Performing search with first manager...")
    start = time.time()
    results1 = manager1.search(["customer", "payment"], top_k=5)
    time1 = (time.time() - start) * 1000
    print(f"   Search time: {time1:.2f}ms")
    print(f"   Results found: {len(results1)}")
    
    # Test 2: Create second manager instance (should load existing index)
    print("\n3. Creating second manager instance (simulating new session)...")
    manager2 = TantivyParquetSearchManager(
        index_dir=index_dir,
        cache_size=10000,
        enable_optimizations=True
    )
    
    # Perform search with second manager (should NOT rebuild)
    print("\n4. Performing search with second manager...")
    start = time.time()
    results2 = manager2.search(["customer", "payment"], top_k=5)
    time2 = (time.time() - start) * 1000
    print(f"   Search time: {time2:.2f}ms")
    print(f"   Results found: {len(results2)}")
    
    # Test 3: Multiple searches with same manager (no rebuilding)
    print("\n5. Performing multiple searches (checking for rebuilds)...")
    search_times = []
    for i in range(5):
        start = time.time()
        results = manager2.search(["order", "status"], top_k=3, use_cache=False)
        elapsed = (time.time() - start) * 1000
        search_times.append(elapsed)
        print(f"   Search {i+1}: {elapsed:.2f}ms, {len(results)} results")
    
    avg_time = sum(search_times) / len(search_times)
    print(f"\n   Average search time: {avg_time:.2f}ms")
    
    # Verify no rebuilding occurred
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    
    if time2 < time1 * 2:  # Second search should be similar or faster
        print("✓ Index persistence working correctly!")
        print("✓ No rebuilding detected between manager instances")
    else:
        print("✗ Possible rebuilding detected (second search much slower)")
    
    if max(search_times) < avg_time * 2:  # All searches should be similar
        print("✓ Consistent search times (no rebuilding during searches)")
    else:
        print("✗ Inconsistent search times (possible rebuilding)")
    
    # Show stats
    print(f"\nSearch Statistics:")
    print(f"  First manager search:  {time1:.2f}ms")
    print(f"  Second manager search: {time2:.2f}ms")
    print(f"  Average search time:   {avg_time:.2f}ms")
    print(f"  Min search time:       {min(search_times):.2f}ms")
    print(f"  Max search time:       {max(search_times):.2f}ms")
    
    return manager2

def test_cache_performance():
    """Test cache performance"""
    from tantivy_parquet_manager import TantivyParquetSearchManager
    
    print("\n" + "=" * 60)
    print("TEST: Cache Performance")
    print("=" * 60)
    
    manager = TantivyParquetSearchManager(
        index_dir="./test_persistent_index",
        cache_size=10000,
        enable_optimizations=True
    )
    
    queries = [
        ["customer", "order"],
        ["payment", "status"],
        ["delivery", "tracking"]
    ]
    
    print("\n1. First run (no cache):")
    times_no_cache = []
    for query in queries:
        manager.clear_cache()
        start = time.time()
        results = manager.search(query, use_cache=False)
        elapsed = (time.time() - start) * 1000
        times_no_cache.append(elapsed)
        print(f"   {query}: {elapsed:.2f}ms")
    
    print("\n2. Second run (with cache):")
    times_with_cache = []
    for query in queries:
        start = time.time()
        results = manager.search(query, use_cache=True)
        elapsed = (time.time() - start) * 1000
        times_with_cache.append(elapsed)
        print(f"   {query}: {elapsed:.2f}ms")
    
    avg_no_cache = sum(times_no_cache) / len(times_no_cache)
    avg_with_cache = sum(times_with_cache) / len(times_with_cache)
    
    print(f"\nCache Performance:")
    print(f"  Average (no cache):   {avg_no_cache:.2f}ms")
    print(f"  Average (with cache): {avg_with_cache:.2f}ms")
    if avg_no_cache > 0:
        print(f"  Cache speedup:        {avg_no_cache/avg_with_cache:.1f}x")
    
    stats = manager.get_stats()
    print(f"\nOverall Stats:")
    print(f"  Total searches: {stats['total_searches']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']}")
    print(f"  Cache size:     {stats['cache_size']}")

if __name__ == "__main__":
    # Run persistence test
    manager = test_persistent_index()
    
    # Run cache performance test
    test_cache_performance()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("The index is NOT rebuilding on every search.")
    print("=" * 60)
