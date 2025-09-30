#!/usr/bin/env python3
"""
Search Script for Querying the Indexed Data
"""

import argparse
import json
import numpy as np
from search_engine import SearchEngine
from embeddings_client import EmbeddingsClient
import time

def parse_column_value_pairs(pairs):
    """Parse column:value pairs from command line"""
    result = {}
    for pair in pairs:
        if ':' not in pair:
            print(f"Warning: Invalid format '{pair}'. Expected 'column:value'")
            continue
        col, val = pair.split(':', 1)
        # Try to parse value type
        if val.lower() == 'null':
            result[col] = None
        elif val.lower() in ['true', 'false']:
            result[col] = val.lower() == 'true'
        else:
            try:
                # Try integer
                result[col] = int(val)
            except ValueError:
                try:
                    # Try float
                    result[col] = float(val)
                except ValueError:
                    # Keep as string
                    result[col] = val
    return result

def main():
    parser = argparse.ArgumentParser(description="Search Engine Query Interface")

    parser.add_argument(
        "--index-path",
        type=str,
        default="indices",
        help="Path to index files"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["dataframe", "keyword", "vector", "hybrid"],
        required=True,
        help="Search mode"
    )

    parser.add_argument(
        "--column-value-pairs",
        type=str,
        nargs="+",
        help="Column:value pairs for dataframe mode (e.g., category_1:category_5 integer_1:100)"
    )

    parser.add_argument(
        "--keywords",
        type=str,
        nargs="+",
        help="Keywords for keyword search"
    )

    parser.add_argument(
        "--text-for-vector",
        type=str,
        help="Text to convert to vector for vector search"
    )

    parser.add_argument(
        "--vector-file",
        type=str,
        help="Path to numpy file containing query vector"
    )

    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum number of results to return"
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark with multiple queries"
    )

    parser.add_argument(
        "--output-format",
        type=str,
        choices=["json", "table", "summary"],
        default="summary",
        help="Output format for results"
    )

    args = parser.parse_args()

    # Initialize search engine
    print(f"Initializing search engine from {args.index_path}...")
    try:
        engine = SearchEngine(
            index_path=args.index_path,
            max_results=args.max_results
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run engine.py first to create indices.")
        return 1

    # Prepare search parameters
    column_value_pairs = None
    keywords = None
    query_vector = None

    # Parse column-value pairs
    if args.column_value_pairs:
        column_value_pairs = parse_column_value_pairs(args.column_value_pairs)

    # Parse keywords
    if args.keywords:
        keywords = args.keywords

    # Prepare vector
    if args.text_for_vector:
        # Convert text to vector using embeddings client
        print("Converting text to vector...")
        client = EmbeddingsClient(
            dimension=4096,  # Should match ingestion settings
            cache_dir=f"{args.index_path}/embeddings_cache"
        )
        query_vector = client.get_embedding(args.text_for_vector)

    elif args.vector_file:
        # Load vector from file
        query_vector = np.load(args.vector_file)

    # Validate parameters for mode
    if args.mode == "dataframe" and not column_value_pairs:
        print("Error: column-value-pairs required for dataframe mode")
        return 1
    elif args.mode == "keyword" and not keywords:
        print("Error: keywords required for keyword mode")
        return 1
    elif args.mode == "vector" and query_vector is None:
        print("Error: text-for-vector or vector-file required for vector mode")
        return 1
    elif args.mode == "hybrid":
        if not any([column_value_pairs, keywords, query_vector is not None]):
            print("Error: At least one search parameter required for hybrid mode")
            return 1

    # Run benchmark if requested
    if args.benchmark:
        print("\nRunning performance benchmark...")
        print("-" * 50)

        num_queries = 100
        total_time = 0
        successful_queries = 0

        for i in range(num_queries):
            # Modify parameters slightly for each query
            test_params = {}

            if column_value_pairs:
                test_params['column_value_pairs'] = column_value_pairs

            if keywords:
                # Rotate keywords
                test_params['keywords'] = keywords[i % len(keywords):] + keywords[:i % len(keywords)]

            if query_vector is not None:
                # Add small noise to vector
                noise = np.random.randn(*query_vector.shape) * 0.01
                test_params['query_vector'] = query_vector + noise

            # Run search
            result = engine.search(args.mode, **test_params)

            # Parse time
            time_str = result.get('time_taken', '0s')
            time_val = float(time_str.replace('s', ''))
            total_time += time_val

            if result['status'] == 'success':
                successful_queries += 1

            if (i + 1) % 10 == 0:
                avg_time = total_time / (i + 1)
                print(f"Completed {i + 1}/{num_queries} queries. Avg time: {avg_time:.3f}s")

        # Print benchmark results
        avg_time = total_time / num_queries
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)
        print(f"Total queries: {num_queries}")
        print(f"Successful queries: {successful_queries}")
        print(f"Average query time: {avg_time:.3f}s")
        print(f"95th percentile target: 0.200s")
        print(f"Target met: {'YES' if avg_time < 0.200 else 'NO'}")
        print("=" * 50)

    else:
        # Run single search
        print(f"\nSearching in {args.mode} mode...")
        print("-" * 50)

        # Execute search
        result = engine.search(
            mode=args.mode,
            column_value_pairs=column_value_pairs,
            keywords=keywords,
            query_vector=query_vector
        )

        # Display results based on format
        if args.output_format == "json":
            # Full JSON output
            print(json.dumps(result, indent=2, default=str))

        elif args.output_format == "table":
            # Table format
            print(f"Status: {result['status']}")
            print(f"Count: {result['count']}")
            print(f"Time: {result['time_taken']}")
            print("\nResults:")
            print("-" * 50)

            for i, res in enumerate(result['results'], 1):
                print(f"\n--- Result {i} ---")
                # Show key fields
                for key in ['id', 'email', 'url', 'category_1', 'integer_1', 'rrf_score']:
                    if key in res:
                        print(f"  {key}: {res[key]}")

                # Show first 100 chars of text fields
                for key in res:
                    if 'text' in key and isinstance(res[key], str):
                        preview = res[key][:100] + "..." if len(res[key]) > 100 else res[key]
                        print(f"  {key}: {preview}")

        else:  # summary format
            print(f"Status: {result['status']}")
            print(f"Results found: {result['count']}")
            print(f"Time taken: {result['time_taken']}")

            if args.mode == "hybrid":
                print(f"Modes used: {', '.join(result.get('modes_used', []))}")

            if result['results']:
                print(f"\nShowing top {len(result['results'])} results:")
                print("-" * 30)

                for i, res in enumerate(result['results'], 1):
                    # Show compact summary
                    id_field = res.get('id', res.get('email', res.get('url', f"Row {i}")))
                    score = res.get('rrf_score', res.get('score', ''))

                    print(f"{i}. {id_field}")
                    if score:
                        print(f"   Score: {score:.4f}" if isinstance(score, float) else f"   Score: {score}")

                    # Show one text preview
                    for key in ['short_text_1', 'medium_text_1', 'long_text_1']:
                        if key in res and res[key]:
                            preview = res[key][:80] + "..." if len(res[key]) > 80 else res[key]
                            print(f"   Preview: {preview}")
                            break

    # Clean up
    engine.close()
    print("\nSearch complete!")

if __name__ == "__main__":
    main()