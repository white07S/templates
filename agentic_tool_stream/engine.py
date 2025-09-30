#!/usr/bin/env python3
"""
Main Engine Script for Data Ingestion and Indexing
"""

import argparse
import os
from dotenv import load_dotenv
from ingestion import IngestionPipeline
from mock_data_generator import MockDataGenerator

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Search Engine Ingestion Pipeline")

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/mock_data.parquet",
        help="Path to parquet data file"
    )

    parser.add_argument(
        "--index-path",
        type=str,
        default="indices",
        help="Path to store index files"
    )

    parser.add_argument(
        "--embedding-columns",
        type=str,
        nargs="+",
        default=["long_text_1", "long_text_2", "medium_text_1", "short_text_1"],
        help="Columns to convert to embeddings (max 4)"
    )

    parser.add_argument(
        "--vector-dimension",
        type=int,
        default=4096,
        help="Dimension of embedding vectors"
    )

    parser.add_argument(
        "--openai-key",
        type=str,
        default=None,
        help="OpenAI API key (defaults to OPENAI_API_KEY env var)"
    )

    parser.add_argument(
        "--openai-url",
        type=str,
        default=None,
        help="OpenAI API URL (defaults to OPENAI_API_URL env var)"
    )

    parser.add_argument(
        "--openai-model",
        type=str,
        default="text-embedding-3-large",
        help="OpenAI embedding model"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for processing"
    )

    parser.add_argument(
        "--re-index",
        action="store_true",
        help="Force re-indexing even if indices exist"
    )

    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate mock data before ingestion"
    )

    parser.add_argument(
        "--num-rows",
        type=int,
        default=1000000,
        help="Number of rows to generate (if --generate-data)"
    )

    args = parser.parse_args()

    # Generate mock data if requested
    if args.generate_data:
        print(f"Generating mock data with {args.num_rows:,} rows...")
        generator = MockDataGenerator(
            num_rows=args.num_rows,
            output_path=args.data_path
        )
        generator.generate()
        print()

    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        print("Use --generate-data to create mock data, or provide a valid --data-path")
        return 1

    # Limit embedding columns to 4
    if len(args.embedding_columns) > 4:
        print("Warning: Maximum 4 embedding columns allowed. Using first 4.")
        args.embedding_columns = args.embedding_columns[:4]

    # Create ingestion pipeline
    pipeline = IngestionPipeline(
        data_path=args.data_path,
        index_path=args.index_path,
        embedding_columns=args.embedding_columns,
        vector_dimension=args.vector_dimension,
        openai_key=args.openai_key,
        openai_url=args.openai_url,
        openai_model=args.openai_model,
        batch_size=args.batch_size,
        re_index=args.re_index
    )

    # Run ingestion
    print("Starting ingestion and indexing pipeline...")
    print("-" * 50)
    pipeline.ingest()

    print("\n" + "=" * 50)
    print("Ingestion complete! You can now run search.py to query the data.")
    print("=" * 50)

if __name__ == "__main__":
    main()