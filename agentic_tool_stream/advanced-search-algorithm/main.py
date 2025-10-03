"""
Main entry point for the advanced search algorithm.
"""
import asyncio
import sys
import argparse
from pathlib import Path
from search import SearchInterface
from ingestion import DataIngestion


async def ingest_data(force_reindex: bool = False):
    """Run data ingestion."""
    print("Starting data ingestion...")
    ingestion = DataIngestion(force_reindex=force_reindex)
    success = await ingestion.ingest_and_index()
    if success:
        print("✓ Ingestion completed successfully!")
    else:
        print("✗ Ingestion failed!")
        sys.exit(1)


async def run_search(query: str, mode: str = "combined", top_k: int = 5):
    """Run a search query."""
    search = SearchInterface(auto_ingest=False)

    # Parse query based on mode
    if mode in ["keyword", "combined"]:
        # Split by comma for multiple queries
        queries = [q.strip() for q in query.split(",")]
    else:
        queries = query

    # Perform search
    results = await search.engine.search(queries, mode=mode, top_k=top_k)

    # Print results
    search.print_results(results)

    # Export results
    output_file = f"{mode}_search_results.json"
    search.export_results(results, output_file)

    return results


async def interactive_mode():
    """Run in interactive mode."""
    print("\n=== Advanced Search Algorithm - Interactive Mode ===\n")

    # Initialize search
    search = SearchInterface(auto_ingest=True)

    while True:
        print("\nOptions:")
        print("1. Keyword Search")
        print("2. Semantic Search")
        print("3. Combined Search")
        print("4. Benchmark")
        print("5. Re-index Data")
        print("6. Exit")

        choice = input("\nEnter choice (1-6): ").strip()

        if choice == "6":
            print("Goodbye!")
            break
        elif choice == "5":
            await ingest_data(force_reindex=True)
            search = SearchInterface(auto_ingest=False)
        elif choice == "4":
            from search import benchmark
            await benchmark()
        elif choice in ["1", "2", "3"]:
            mode_map = {"1": "keyword", "2": "semantic", "3": "combined"}
            mode = mode_map[choice]

            if mode == "keyword":
                print("Enter search queries (comma-separated for multiple):")
            elif mode == "semantic":
                print("Enter search query:")
            else:
                print("Enter search queries (comma-separated for multiple):")

            query = input("> ").strip()
            if not query:
                print("No query entered!")
                continue

            top_k = input("Number of results (default 5): ").strip()
            top_k = int(top_k) if top_k else 5

            await run_search(query, mode=mode, top_k=top_k)
        else:
            print("Invalid choice!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Advanced Search Algorithm")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest and index data")
    ingest_parser.add_argument("--force", action="store_true", help="Force re-indexing")

    # Search command
    search_parser = subparsers.add_parser("search", help="Perform a search")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--mode", choices=["keyword", "semantic", "combined"],
                              default="combined", help="Search mode")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run performance benchmark")

    # Interactive mode (default)
    interactive_parser = subparsers.add_parser("interactive", help="Run in interactive mode")

    args = parser.parse_args()

    if args.command == "ingest":
        asyncio.run(ingest_data(force_reindex=args.force))
    elif args.command == "search":
        asyncio.run(run_search(args.query, mode=args.mode, top_k=args.top_k))
    elif args.command == "benchmark":
        from search import benchmark
        asyncio.run(benchmark())
    else:
        # Default to interactive mode
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()
