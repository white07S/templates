"""
Quick test script for JSON search system
"""

import os
import orjson
import time
import config

# Reduce data count for quick test
config.MOCK_DATA_COUNT = 100

def create_test_data():
    """Create a small set of test data"""
    os.makedirs(config.DATA_DIR, exist_ok=True)

    print(f"Creating {config.MOCK_DATA_COUNT} test JSON files...")

    for i in range(config.MOCK_DATA_COUNT):
        filename = f"test_{i:04d}.json"
        data = {
            "id": i,
            "title": f"Document {i}",
            "content": f"This is test document number {i}",
            "nested": {
                "deep": {
                    "value": f"Deep value {i}",
                    "list": [f"item_{j}" for j in range(5)]
                }
            },
            "tags": [f"tag_{j}" for j in range(3)]
        }

        # Add some specific content to certain documents
        if i % 10 == 0:
            data["content"] += " important document"
        if i % 15 == 0:
            data["content"] += " critical system"
        if i % 20 == 0:
            data["content"] += " user authentication"

        filepath = os.path.join(config.DATA_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    print(f"Created {config.MOCK_DATA_COUNT} test files")


def test_indexing():
    """Test indexing functionality"""
    from indexer import JSONIndexer

    print("\nTesting indexing...")
    indexer = JSONIndexer()
    indexer.clear_index()

    start_time = time.time()
    indexed, errors = indexer.index_directory()
    end_time = time.time()

    print(f"Indexed {indexed} documents in {end_time - start_time:.2f} seconds")
    print(f"Errors: {errors}")

    stats = indexer.get_index_stats()
    print(f"Total documents in index: {stats['num_documents']}")

    return indexed > 0


def test_search():
    """Test search functionality"""
    from searcher import JSONSearcher, format_search_results

    print("\nTesting search...")
    searcher = JSONSearcher()

    test_queries = [
        "test_0010",
        "important document",
        "critical system",
        "Deep value 42",
        "test",  # Test partial filename matching
    ]

    for query in test_queries:
        print(f"\n{'-'*40}")
        print(f"Query: '{query}'")

        start_time = time.time()
        results = searcher.search(query)
        end_time = time.time()

        print(f"Search time: {end_time - start_time:.3f} seconds")
        print(f"Results found: {len(results)}")

        if results:
            print(f"Top result: {results[0]['filename']} (score: {results[0]['rrf_score']:.4f})")


def main():
    """Run quick tests"""
    print("Quick Test for JSON Search System")
    print("=" * 60)

    # Create test data
    create_test_data()

    # Test indexing
    if test_indexing():
        # Test searching
        test_search()
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
    else:
        print("Indexing failed!")


if __name__ == "__main__":
    main()