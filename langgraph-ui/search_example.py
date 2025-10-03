"""
Example usage of the advanced search engine.
"""
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import time
import os

# Set environment variables before importing modules
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"  # Replace with your key
os.environ["DATA_PATH"] = "data/sample.parquet"


async def create_sample_data():
    """Create a sample parquet file for testing."""
    print("Creating sample data...")
    
    # Create sample dataframe with various column types
    np.random.seed(42)
    n_rows = 10000
    
    data = {
        # Text columns
        "title": [f"Product {i}: {np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'])}" 
                 for i in range(n_rows)],
        "description": [f"This is a {np.random.choice(['great', 'amazing', 'wonderful', 'fantastic'])} "
                       f"{np.random.choice(['item', 'product', 'choice', 'option'])} for "
                       f"{np.random.choice(['daily use', 'special occasions', 'professionals', 'everyone'])}"
                       for _ in range(n_rows)],
        "category": np.random.choice(['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports'], n_rows),
        
        # Numeric columns
        "price": np.random.uniform(10, 1000, n_rows).round(2),
        "rating": np.random.uniform(1, 5, n_rows).round(1),
        "stock": np.random.randint(0, 100, n_rows),
        
        # Date column
        "created_date": pd.date_range('2023-01-01', periods=n_rows, freq='H'),
        
        # Columns with edge cases
        "notes": [np.nan if i % 10 == 0 else f"Note {i}" for i in range(n_rows)],
        "tags": ["" if i % 15 == 0 else f"tag{i % 5}, tag{i % 7}" for i in range(n_rows)],
        
        # Mixed types
        "sku": [f"SKU-{i:05d}" for i in range(n_rows)],
        "vendor_id": [None if i % 20 == 0 else f"VENDOR-{i % 50}" for i in range(n_rows)]
    }
    
    df = pd.DataFrame(data)
    
    # Add some completely null columns to test edge cases
    df['null_column'] = None
    df['empty_column'] = ""
    
    # Save to parquet
    Path("data").mkdir(exist_ok=True)
    df.to_parquet("data/sample.parquet", index=True)
    
    print(f"Created sample data with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {', '.join(df.columns)}")
    print(f"File size: {os.path.getsize('data/sample.parquet') / 1024 / 1024:.2f} MB")
    
    return df


async def run_ingestion():
    """Run the ingestion process."""
    from ingestion import DataIngestion
    from config import SearchConfig
    
    print("\n" + "="*50)
    print("Starting Ingestion Process")
    print("="*50)
    
    # Update config for sample data
    SearchConfig.DATA_PATH = "data/sample.parquet"
    SearchConfig.EMBEDDINGS_COLUMNS = ["title", "description"]
    SearchConfig.REINDEX = True
    SearchConfig.CHUNK_SIZE = 500
    
    # Run ingestion
    ingestion = DataIngestion()
    await ingestion.ingest_data()
    
    print("\nIngestion completed!")


async def run_search_examples():
    """Run various search examples."""
    from search import SearchAPI
    from config import SearchConfig
    
    print("\n" + "="*50)
    print("Running Search Examples")
    print("="*50)
    
    # Initialize search API
    api = SearchAPI()
    
    # Example 1: Keyword Search
    print("\n1. KEYWORD SEARCH")
    print("-" * 30)
    queries = ["Electronics", "amazing product"]
    response = await api.search(queries, mode="keyword", top_k=5)
    print(f"Query: {queries}")
    print(f"Found {response.count} results in {response.search_time*1000:.1f}ms")
    if response.results:
        for i, result in enumerate(response.results[:3], 1):
            print(f"  {i}. {result.get('title', 'N/A')} (Score: {result['_score']:.3f})")
    
    # Example 2: Semantic Search
    print("\n2. SEMANTIC SEARCH")
    print("-" * 30)
    query = "high quality items for professional use"
    response = await api.search(query, mode="semantic", top_k=5)
    print(f"Query: '{query}'")
    print(f"Found {response.count} results in {response.search_time*1000:.1f}ms")
    if response.results:
        for i, result in enumerate(response.results[:3], 1):
            print(f"  {i}. {result.get('title', 'N/A')} (Score: {result['_score']:.3f})")
    
    # Example 3: Hybrid Search with RRF
    print("\n3. HYBRID SEARCH (RRF)")
    print("-" * 30)
    query = "Sports equipment"
    response = await api.search(query, mode="hybrid", top_k=5)
    print(f"Query: '{query}'")
    print(f"Found {response.count} results in {response.search_time*1000:.1f}ms")
    if response.results:
        for i, result in enumerate(response.results[:3], 1):
            print(f"  {i}. {result.get('title', 'N/A')} (Score: {result['_score']:.3f})")
    
    # Example 4: Field-specific search
    print("\n4. FIELD-SPECIFIC KEYWORD SEARCH")
    print("-" * 30)
    response = await api.search(
        "Books", 
        mode="keyword", 
        top_k=5,
        fields=["category"]
    )
    print(f"Query: 'Books' in field 'category'")
    print(f"Found {response.count} results in {response.search_time*1000:.1f}ms")
    
    # Example 5: Weighted Hybrid Search
    print("\n5. WEIGHTED HYBRID SEARCH")
    print("-" * 30)
    query = "professional electronics"
    response = await api.search(
        query,
        mode="hybrid",
        top_k=5,
        keyword_weight=0.3,
        semantic_weight=0.7  # Favor semantic results
    )
    print(f"Query: '{query}' (semantic weight: 0.7, keyword weight: 0.3)")
    print(f"Found {response.count} results in {response.search_time*1000:.1f}ms")
    
    # Example 6: Batch Search
    print("\n6. BATCH SEARCH")
    print("-" * 30)
    queries = ["Electronics", "Clothing", "Sports equipment"]
    start_time = time.time()
    responses = await api.batch_search(queries, mode="hybrid", top_k=3)
    batch_time = time.time() - start_time
    print(f"Processed {len(queries)} queries in {batch_time*1000:.1f}ms total")
    for query, response in zip(queries, responses):
        print(f"  '{query}': {response.count} results in {response.search_time*1000:.1f}ms")


async def benchmark_performance():
    """Run performance benchmarks."""
    from search import SearchAPI
    import statistics
    
    print("\n" + "="*50)
    print("Performance Benchmark")
    print("="*50)
    
    api = SearchAPI()
    
    # Test queries
    test_queries = [
        "Electronics",
        "professional equipment",
        "high quality products",
        "amazing items for daily use",
        "Sports and outdoor gear"
    ]
    
    modes = ["keyword", "semantic", "hybrid"]
    
    for mode in modes:
        print(f"\n{mode.upper()} SEARCH BENCHMARK")
        print("-" * 30)
        
        times = []
        for _ in range(10):  # Run each query 10 times
            for query in test_queries:
                response = await api.search(query, mode=mode, top_k=10)
                times.append(response.search_time * 1000)  # Convert to ms
        
        print(f"Average: {statistics.mean(times):.1f}ms")
        print(f"Median:  {statistics.median(times):.1f}ms")
        print(f"Min:     {min(times):.1f}ms")
        print(f"Max:     {max(times):.1f}ms")
        print(f"StdDev:  {statistics.stdev(times):.1f}ms")
        
        # Check against target
        if statistics.median(times) < 150:
            print(f"✓ Meets <150ms target!")
        else:
            print(f"✗ Exceeds 150ms target")


async def main():
    """Main example runner."""
    print("ADVANCED SEARCH ENGINE EXAMPLE")
    print("="*50)
    
    # Check if sample data exists
    if not Path("data/sample.parquet").exists():
        await create_sample_data()
    
    # Check if indices exist
    from config import SearchConfig
    SearchConfig.DATA_PATH = "data/sample.parquet"
    
    indices_exist = (
        Path(SearchConfig.TANTIVY_INDEX_PATH).exists() and
        Path(SearchConfig.FAISS_INDEX_PATH).exists()
    )
    
    if not indices_exist:
        print("\nIndices not found. Running ingestion...")
        await run_ingestion()
    else:
        print("\nIndices found. Skipping ingestion.")
        print("To rebuild indices, delete the 'data/indices' folder.")
    
    # Run search examples
    await run_search_examples()
    
    # Run benchmarks
    await benchmark_performance()
    
    print("\n" + "="*50)
    print("Example completed!")
    print("="*50)


if __name__ == "__main__":
    # Note: Make sure to set your OPENAI_API_KEY in the environment or in this file
    asyncio.run(main())