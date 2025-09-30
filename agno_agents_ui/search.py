"""Main search interface for the search engine."""

import json
import numpy as np
import argparse
from typing import Dict, Any, List, Optional

from src.core.models import SearchMode, SearchRequest, SearchResult
from src.core.config import SearchConfig
from src.search.search_engine import SearchEngine


def parse_column_value_pairs(pairs: List[str]) -> Dict[str, Any]:
    """Parse column-value pairs from command line arguments.

    Args:
        pairs: List of "column=value" strings

    Returns:
        Dictionary of column-value pairs
    """
    result = {}
    for pair in pairs:
        if '=' not in pair:
            raise ValueError(f"Invalid column-value pair: {pair}")
        column, value = pair.split('=', 1)

        # Try to parse value as different types
        try:
            # Try integer
            result[column] = int(value)
        except ValueError:
            try:
                # Try float
                result[column] = float(value)
            except ValueError:
                # Try boolean
                if value.lower() in ['true', 'false']:
                    result[column] = value.lower() == 'true'
                else:
                    # Keep as string
                    result[column] = value

    return result


def print_results(result: SearchResult, verbose: bool = False):
    """Print search results in a formatted way.

    Args:
        result: Search result object
        verbose: Whether to print full documents
    """
    print("\n" + "=" * 80)
    print("SEARCH RESULTS")
    print("=" * 80)

    print(f"\nStatus: {result.status}")
    print(f"Total matches: {result.count}")
    print(f"Time taken: {result.time_taken:.3f} seconds ({result.time_taken * 1000:.1f} ms)")

    if result.results:
        print(f"\nShowing top {len(result.results)} results:")
        print("-" * 80)

        for i, doc in enumerate(result.results, 1):
            print(f"\nResult #{i}:")

            if verbose:
                # Print full document
                for key, value in doc.items():
                    if key.startswith('_'):
                        continue  # Skip internal fields
                    print(f"  {key}: {value}")
            else:
                # Print summary (selected fields only)
                summary_fields = ['id', 'title', 'category', 'status', 'author', '_rrf_score']
                for field in summary_fields:
                    if field in doc:
                        value = doc[field]
                        if isinstance(value, str) and len(value) > 100:
                            value = value[:97] + "..."
                        print(f"  {field}: {value}")

            if i < len(result.results):
                print("-" * 40)

    print("\n" + "=" * 80)


def main():
    """Main entry point for search interface."""
    parser = argparse.ArgumentParser(description="Search engine interface")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["dataframe", "keyword", "vector", "hybrid"],
        required=True,
        help="Search mode to use"
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="data/index",
        help="Path to index files"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to return"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full document details"
    )

    # Mode-specific arguments
    parser.add_argument(
        "--column-value",
        type=str,
        nargs="+",
        help="Column-value pairs for dataframe mode (format: column=value)"
    )
    parser.add_argument(
        "--keywords",
        type=str,
        nargs="+",
        help="Keywords for keyword search mode"
    )
    parser.add_argument(
        "--vector-random",
        action="store_true",
        help="Use a random vector for vector search (for testing)"
    )
    parser.add_argument(
        "--vector-file",
        type=str,
        help="Path to file containing vector (numpy array)"
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

    # Prepare search request
    request_params = {
        "mode": SearchMode(args.mode),
        "top_k": args.top_k
    }

    # Add mode-specific parameters
    if args.mode == "dataframe":
        if not args.column_value:
            parser.error("--column-value required for dataframe mode")
        request_params["column_value_pairs"] = parse_column_value_pairs(args.column_value)

    elif args.mode == "keyword":
        if not args.keywords:
            parser.error("--keywords required for keyword mode")
        request_params["keywords"] = args.keywords

    elif args.mode == "vector":
        if args.vector_random:
            # Generate random vector for testing
            request_params["vector"] = np.random.randn(4096).tolist()
        elif args.vector_file:
            # Load vector from file
            vector = np.load(args.vector_file)
            request_params["vector"] = vector.tolist()
        else:
            parser.error("Either --vector-random or --vector-file required for vector mode")

    elif args.mode == "hybrid":
        # Hybrid mode can use any combination
        if args.column_value:
            request_params["column_value_pairs"] = parse_column_value_pairs(args.column_value)
        if args.keywords:
            request_params["keywords"] = args.keywords
        if args.vector_random:
            request_params["vector"] = np.random.randn(4096).tolist()
        elif args.vector_file:
            vector = np.load(args.vector_file)
            request_params["vector"] = vector.tolist()

        # Check at least one parameter is provided
        if not any([args.column_value, args.keywords, args.vector_random, args.vector_file]):
            parser.error("At least one search parameter required for hybrid mode")

    # Create search request
    request = SearchRequest(**request_params)

    # Perform search
    print(f"\nPerforming {args.mode} search...")
    result = engine.search(request)

    # Print results
    print_results(result, verbose=args.verbose)


if __name__ == "__main__":
    main()