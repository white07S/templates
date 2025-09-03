"""
DuckDB JSON Full-Text Search System
A dynamic, performant solution for indexing and searching JSON columns in DuckDB files.

Requirements:
- pip install duckdb
"""

import duckdb
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


class DuckDBJSONSearch:
    """
    A high-performance full-text search system for DuckDB files with JSON columns.
    Dynamically handles any schema with a 'hash' primary key and JSON columns.
    """
    
    def __init__(self, db_path: str, temp_db: Optional[str] = None):
        """
        Initialize the search system.
        
        Args:
            db_path: Path to the DuckDB file to search
            temp_db: Optional path for temporary index database (default: in-memory)
        """
        self.db_path = db_path
        self.temp_db = temp_db or ':memory:'
        self.conn = None
        self.source_conn = None
        self.table_name = None
        self.json_columns = []
        self.index_created = False
        
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        
    def connect(self):
        """Establish connections to source and index databases."""
        # Connection for the index database
        self.conn = duckdb.connect(self.temp_db)
        
        # Install and load FTS extension
        self.conn.execute("""
            INSTALL fts;
            LOAD fts;
        """)
        
        # Connection to source database
        self.source_conn = duckdb.connect(self.db_path, read_only=True)
        
    def discover_schema(self, table_name: str) -> Dict[str, str]:
        """
        Discover the schema of the table and identify JSON columns.
        
        Args:
            table_name: Name of the table to analyze
            
        Returns:
            Dictionary of column names and their types
        """
        self.table_name = table_name
        
        # Get table schema
        schema = self.source_conn.execute(f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
        """).fetchall()
        
        if not schema:
            raise ValueError(f"Table '{table_name}' not found in database")
        
        columns = {}
        for col_name, col_type in schema:
            columns[col_name] = col_type
            # Identify JSON columns (could be JSON or VARCHAR containing JSON)
            if col_type in ['JSON', 'VARCHAR']:
                if col_name != 'hash':
                    self.json_columns.append(col_name)
        
        # Verify hash column exists
        if 'hash' not in columns:
            raise ValueError("Table must have a 'hash' column as primary key")
            
        if not self.json_columns:
            raise ValueError("No JSON columns found in table")
            
        print(f"✓ Found table '{table_name}' with columns: {list(columns.keys())}")
        print(f"✓ JSON columns identified: {self.json_columns}")
        
        return columns
    
    def create_searchable_view(self) -> None:
        """
        Create a view that extracts all JSON values into a searchable format.
        Uses advanced DuckDB JSON functions for optimal performance.
        """
        print("\n⏳ Creating searchable view...")
        start_time = time.time()
        
        # Attach source database
        self.conn.execute(f"ATTACH '{self.db_path}' AS source_db (READ_ONLY)")
        
        # Build dynamic SQL to extract all JSON key-value pairs
        json_extractions = []
        
        for col in self.json_columns:
            # For each JSON column, extract all values recursively
            extraction = f"""
                -- Extract values from {col}
                SELECT 
                    hash,
                    '{col}' as column_name,
                    key,
                    CASE 
                        WHEN json_type(value) = 'VARCHAR' THEN json_extract_string(value, '$')
                        WHEN json_type(value) IN ('BIGINT', 'DOUBLE', 'BOOLEAN') THEN CAST(value AS VARCHAR)
                        ELSE NULL
                    END as text_value
                FROM (
                    SELECT 
                        hash,
                        unnest(json_keys(CASE 
                            WHEN json_valid({col}) THEN {col}::JSON
                            ELSE TRY_CAST({col} AS JSON)
                        END)) as key,
                        json_extract(CASE 
                            WHEN json_valid({col}) THEN {col}::JSON
                            ELSE TRY_CAST({col} AS JSON)
                        END, '$.' || key) as value
                    FROM source_db.{self.table_name}
                    WHERE {col} IS NOT NULL
                )
                WHERE value IS NOT NULL
            """
            json_extractions.append(extraction)
        
        # Combine all extractions with UNION ALL
        union_query = " UNION ALL ".join(json_extractions)
        
        # Create the searchable view
        self.conn.execute(f"""
            CREATE OR REPLACE VIEW searchable_data AS
            WITH extracted_values AS (
                {union_query}
            )
            SELECT 
                hash,
                column_name,
                key,
                text_value,
                -- Combine for full-text search
                column_name || ' ' || key || ' ' || COALESCE(text_value, '') as search_text
            FROM extracted_values
            WHERE text_value IS NOT NULL AND LENGTH(text_value) > 0
        """)
        
        # Create aggregated view for FTS indexing
        self.conn.execute("""
            CREATE OR REPLACE VIEW fts_documents AS
            SELECT 
                hash as doc_id,
                STRING_AGG(search_text, ' ') as content,
                -- Store metadata as JSON for result enrichment
                JSON_OBJECT(
                    'columns', LIST(DISTINCT column_name),
                    'keys', LIST(DISTINCT key),
                    'sample_values', LIST(DISTINCT text_value)[:5]
                ) as metadata
            FROM searchable_data
            GROUP BY hash
        """)
        
        elapsed = time.time() - start_time
        doc_count = self.conn.execute("SELECT COUNT(*) FROM fts_documents").fetchone()[0]
        print(f"✓ Searchable view created in {elapsed:.2f}s")
        print(f"✓ Documents prepared for indexing: {doc_count}")
    
    def create_fts_index(self) -> None:
        """
        Create the full-text search index using DuckDB's FTS extension.
        Optimized for performance with custom settings.
        """
        print("\n⏳ Building FTS index...")
        start_time = time.time()
        
        # Drop existing index if any
        self.conn.execute("""
            BEGIN;
            DROP SCHEMA IF EXISTS fts_main_fts_documents CASCADE;
            COMMIT;
        """)
        
        # Create optimized FTS index
        self.conn.execute("""
            PRAGMA create_fts_index(
                'fts_documents',      -- table name
                'doc_id',            -- unique identifier
                'content',           -- column to index
                overwrite = 1,       -- overwrite if exists
                stemmer = 'porter',  -- use Porter stemmer for better matching
                stopwords = 'english', -- remove common English stopwords
                ignore = '(\\.|[^a-zA-Z0-9])+', -- ignore special characters
                strip_accents = 1,   -- normalize accented characters
                lower = 1            -- case-insensitive search
            )
        """)
        
        self.index_created = True
        elapsed = time.time() - start_time
        
        # Get index statistics
        stats = self.conn.execute("""
            SELECT 
                COUNT(DISTINCT term) as unique_terms,
                COUNT(*) as total_terms
            FROM fts_main_fts_documents.terms
        """).fetchone()
        
        print(f"✓ FTS index created in {elapsed:.2f}s")
        print(f"✓ Unique terms indexed: {stats[0]:,}")
        print(f"✓ Total terms indexed: {stats[1]:,}")
    
    def search(self, keywords: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents matching the given keywords.
        
        Args:
            keywords: List of keywords to search
            top_k: Number of top results to return (default: 5)
            
        Returns:
            List of matching documents with scores and metadata
        """
        if not self.index_created:
            raise RuntimeError("Index not created. Call prepare_index() first.")
        
        # Prepare search query
        search_query = ' '.join(keywords)
        print(f"\n🔍 Searching for: '{search_query}'")
        
        start_time = time.time()
        
        # Perform BM25 search with scoring
        results = self.conn.execute(f"""
            WITH search_results AS (
                SELECT 
                    doc_id as hash,
                    fts_main_fts_documents.match_bm25(
                        doc_id, 
                        ?,
                        k = 1.2,  -- BM25 k parameter (term frequency saturation)
                        b = 0.75  -- BM25 b parameter (length normalization)
                    ) as score,
                    metadata,
                    -- Get snippet for context
                    SUBSTR(content, 1, 200) as snippet
                FROM fts_documents
                WHERE score IS NOT NULL
            ),
            ranked_results AS (
                SELECT 
                    hash,
                    score,
                    metadata,
                    snippet,
                    ROW_NUMBER() OVER (ORDER BY score DESC) as rank
                FROM search_results
            )
            SELECT 
                rank,
                hash,
                ROUND(score, 4) as score,
                JSON_EXTRACT(metadata, '$.columns') as matching_columns,
                JSON_EXTRACT(metadata, '$.keys') as matching_keys,
                snippet
            FROM ranked_results
            WHERE rank <= ?
            ORDER BY rank
        """, [search_query, top_k]).fetchall()
        
        elapsed = time.time() - start_time
        
        # Format results
        formatted_results = []
        for rank, hash_val, score, columns, keys, snippet in results:
            formatted_results.append({
                'rank': rank,
                'hash': hash_val,
                'score': float(score),
                'matching_columns': json.loads(columns) if columns else [],
                'matching_keys': json.loads(keys) if keys else [],
                'snippet': snippet[:150] + '...' if len(snippet) > 150 else snippet
            })
        
        print(f"✓ Search completed in {elapsed*1000:.2f}ms")
        print(f"✓ Found {len(formatted_results)} matching documents")
        
        return formatted_results
    
    def get_full_document(self, hash_value: str) -> Dict[str, Any]:
        """
        Retrieve the full document for a given hash.
        
        Args:
            hash_value: The hash ID of the document
            
        Returns:
            Full document with all JSON columns
        """
        columns_str = ', '.join(self.json_columns)
        result = self.source_conn.execute(f"""
            SELECT hash, {columns_str}
            FROM {self.table_name}
            WHERE hash = ?
        """, [hash_value]).fetchone()
        
        if result:
            doc = {'hash': result[0]}
            for i, col in enumerate(self.json_columns, 1):
                try:
                    doc[col] = json.loads(result[i]) if result[i] else None
                except:
                    doc[col] = result[i]
            return doc
        return None
    
    def prepare_index(self, table_name: str) -> None:
        """
        Complete index preparation pipeline.
        
        Args:
            table_name: Name of the table to index
        """
        print(f"\n{'='*60}")
        print(f"DuckDB JSON Full-Text Search - Index Preparation")
        print(f"{'='*60}")
        
        total_start = time.time()
        
        # 1. Discover schema
        self.discover_schema(table_name)
        
        # 2. Create searchable view
        self.create_searchable_view()
        
        # 3. Create FTS index
        self.create_fts_index()
        
        total_elapsed = time.time() - total_start
        print(f"\n✅ Index preparation completed in {total_elapsed:.2f}s")
        print(f"{'='*60}\n")
    
    def close(self):
        """Close all database connections."""
        if self.conn:
            self.conn.close()
        if self.source_conn:
            self.source_conn.close()


def example_usage():
    """
    Example usage of the DuckDB JSON Search system.
    """
    # Create sample data
    sample_db = "sample_data.duckdb"
    
    # Create a sample database with JSON columns
    with duckdb.connect(sample_db) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                hash VARCHAR PRIMARY KEY,
                product_info JSON,
                specifications JSON,
                metadata JSON
            )
        """)
        
        # Insert sample data
        conn.execute("""
            INSERT INTO products VALUES
            ('hash001', 
             '{"name": "Laptop Pro", "brand": "TechCorp", "category": "Electronics", "price": 1299}',
             '{"cpu": "Intel i7", "ram": "16GB", "storage": "512GB SSD", "display": "15.6 inch"}',
             '{"tags": ["portable", "powerful", "professional"], "rating": 4.5}'),
            ('hash002',
             '{"name": "Wireless Mouse", "brand": "TechCorp", "category": "Accessories", "price": 29}',
             '{"connectivity": "Bluetooth", "battery": "AA", "dpi": "1600", "buttons": 3}',
             '{"tags": ["wireless", "ergonomic", "portable"], "rating": 4.2}'),
            ('hash003',
             '{"name": "Gaming Keyboard", "brand": "GameGear", "category": "Accessories", "price": 89}',
             '{"switches": "Mechanical", "backlight": "RGB", "layout": "Full", "connectivity": "USB"}',
             '{"tags": ["gaming", "mechanical", "rgb"], "rating": 4.8}'),
            ('hash004',
             '{"name": "4K Monitor", "brand": "ViewTech", "category": "Displays", "price": 499}',
             '{"resolution": "3840x2160", "refresh": "60Hz", "panel": "IPS", "size": "27 inch"}',
             '{"tags": ["4k", "professional", "color-accurate"], "rating": 4.6}'),
            ('hash005',
             '{"name": "USB Hub", "brand": "TechCorp", "category": "Accessories", "price": 39}',
             '{"ports": "4 USB 3.0", "power": "External", "cable": "1m", "material": "Aluminum"}',
             '{"tags": ["usb", "portable", "hub"], "rating": 4.0}')
        """)
        print(f"✓ Sample database created: {sample_db}")
    
    # Use the search system
    with DuckDBJSONSearch(sample_db) as searcher:
        # Prepare the index
        searcher.prepare_index('products')
        
        # Example searches
        test_searches = [
            ['laptop', 'portable'],
            ['TechCorp'],
            ['gaming', 'rgb'],
            ['4k', 'monitor'],
            ['wireless', 'bluetooth']
        ]
        
        for keywords in test_searches:
            print(f"\n{'='*60}")
            results = searcher.search(keywords, top_k=3)
            
            for result in results:
                print(f"\nRank #{result['rank']} | Score: {result['score']}")
                print(f"Hash: {result['hash']}")
                print(f"Matching columns: {', '.join(result['matching_columns'])}")
                print(f"Matching keys: {', '.join(result['matching_keys'][:5])}")
                print(f"Snippet: {result['snippet']}")
                
                # Optionally get full document
                if result['rank'] == 1:
                    full_doc = searcher.get_full_document(result['hash'])
                    print(f"\nFull document for top result:")
                    print(json.dumps(full_doc, indent=2)[:500] + '...')


if __name__ == "__main__":
    example_usage()

#!/usr/bin/env python3
"""
DuckDB JSON Full-Text Search CLI
Production-ready command-line interface for searching JSON data in DuckDB files.

Usage:
    python duckdb_json_search_cli.py index <db_file> <table_name> [--index-db <path>]
    python duckdb_json_search_cli.py search <db_file> <table_name> <keywords> [--index-db <path>]
    python duckdb_json_search_cli.py batch-search <db_file> <table_name> <keywords_file> [--index-db <path>]
"""

import argparse
import duckdb
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any


class OptimizedJSONSearcher:
    """Optimized version focusing on performance."""
    
    def __init__(self, db_path: str, index_path: str = ':memory:'):
        self.db_path = db_path
        self.index_path = index_path
        self.conn = duckdb.connect(index_path)
        
        # Configure for performance
        self.conn.execute("""
            SET memory_limit = '4GB';
            SET threads = 4;
            INSTALL fts;
            LOAD fts;
        """)
        
    def build_index(self, table_name: str) -> Dict[str, Any]:
        """Build FTS index with performance optimizations."""
        start_time = time.time()
        stats = {}
        
        # Attach source database
        self.conn.execute(f"ATTACH '{self.db_path}' AS src (READ_ONLY)")
        
        # Get JSON columns
        columns = self.conn.execute(f"""
            SELECT column_name 
            FROM src.information_schema.columns 
            WHERE table_name = '{table_name}' 
                AND data_type IN ('JSON', 'VARCHAR')
                AND column_name != 'hash'
        """).fetchall()
        
        json_cols = [col[0] for col in columns]
        stats['json_columns'] = json_cols
        
        if not json_cols:
            raise ValueError("No JSON columns found")
        
        # Build efficient extraction query using json_each for better performance
        extractions = []
        for col in json_cols:
            extractions.append(f"""
                SELECT 
                    t.hash,
                    '{col}' || '.' || e.key || ':' || e.value as text_fragment
                FROM src.{table_name} t,
                    json_each(
                        CASE 
                            WHEN json_valid(t.{col}) THEN t.{col}::JSON
                            ELSE TRY_CAST(t.{col} AS JSON)
                        END
                    ) e
                WHERE t.{col} IS NOT NULL
                    AND e.value IS NOT NULL
                    AND json_type(e.value) IN ('VARCHAR', 'BIGINT', 'DOUBLE', 'BOOLEAN')
            """)
        
        # Create materialized table for better performance
        union_query = " UNION ALL ".join(extractions)
        
        self.conn.execute(f"""
            CREATE TABLE fts_data AS
            SELECT 
                hash,
                STRING_AGG(text_fragment, ' ') as content
            FROM ({union_query}) fragments
            GROUP BY hash
        """)
        
        stats['total_documents'] = self.conn.execute(
            "SELECT COUNT(*) FROM fts_data"
        ).fetchone()[0]
        
        # Create FTS index with optimized settings
        self.conn.execute("""
            PRAGMA create_fts_index(
                'fts_data',
                'hash',
                'content',
                overwrite = 1,
                stemmer = 'porter',
                stopwords = 'english',
                ignore = '(\\.|[^a-zA-Z0-9])+',
                strip_accents = 1,
                lower = 1
            )
        """)
        
        stats['index_time'] = time.time() - start_time
        stats['index_size_mb'] = self._get_index_size()
        
        # Save index metadata
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS index_metadata (
                table_name VARCHAR,
                db_path VARCHAR,
                json_columns JSON,
                doc_count INTEGER,
                created_at TIMESTAMP
            )
        """)
        
        self.conn.execute("""
            INSERT INTO index_metadata VALUES (?, ?, ?, ?, NOW())
        """, [table_name, self.db_path, json.dumps(json_cols), stats['total_documents']])
        
        return stats
    
    def search(self, keywords: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """Fast search using BM25 scoring."""
        query = ' '.join(keywords)
        
        results = self.conn.execute(f"""
            SELECT 
                hash,
                ROUND(fts_main_fts_data.match_bm25(hash, ?, k=1.2, b=0.75), 4) as score,
                SUBSTR(content, 1, 150) as snippet
            FROM fts_data
            WHERE score IS NOT NULL
            ORDER BY score DESC
            LIMIT ?
        """, [query, limit]).fetchall()
        
        return [
            {
                'hash': r[0],
                'score': float(r[1]),
                'snippet': r[2]
            }
            for r in results
        ]
    
    def batch_search(self, queries: List[List[str]], limit: int = 5) -> Dict[str, List]:
        """Batch search for multiple queries efficiently."""
        results = {}
        for query_keywords in queries:
            query_key = ' '.join(query_keywords)
            results[query_key] = self.search(query_keywords, limit)
        return results
    
    def _get_index_size(self) -> float:
        """Get index size in MB."""
        try:
            size = self.conn.execute("""
                SELECT 
                    SUM(estimated_size) / (1024.0 * 1024.0) as size_mb
                FROM duckdb_tables()
                WHERE schema_name = 'fts_main_fts_data'
            """).fetchone()
            return round(size[0] if size and size[0] else 0, 2)
        except:
            return 0.0


def main():
    parser = argparse.ArgumentParser(
        description='DuckDB JSON Full-Text Search CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index for a table
  %(prog)s index mydata.duckdb products --index-db ./search_index.duckdb
  
  # Search for keywords
  %(prog)s search mydata.duckdb products "laptop wireless" --index-db ./search_index.duckdb
  
  # Batch search from file
  %(prog)s batch-search mydata.duckdb products keywords.txt --index-db ./search_index.duckdb
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Build search index')
    index_parser.add_argument('db_file', help='DuckDB database file')
    index_parser.add_argument('table', help='Table name')
    index_parser.add_argument('--index-db', default=':memory:', 
                             help='Index database path (default: in-memory)')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for keywords')
    search_parser.add_argument('db_file', help='DuckDB database file')
    search_parser.add_argument('table', help='Table name')
    search_parser.add_argument('keywords', help='Keywords to search (space-separated)')
    search_parser.add_argument('--index-db', default=':memory:',
                              help='Index database path')
    search_parser.add_argument('--limit', type=int, default=5,
                              help='Number of results (default: 5)')
    search_parser.add_argument('--json', action='store_true',
                              help='Output as JSON')
    
    # Batch search command
    batch_parser = subparsers.add_parser('batch-search', 
                                         help='Batch search from file')
    batch_parser.add_argument('db_file', help='DuckDB database file')
    batch_parser.add_argument('table', help='Table name')
    batch_parser.add_argument('keywords_file', help='File with keywords (one query per line)')
    batch_parser.add_argument('--index-db', default=':memory:',
                             help='Index database path')
    batch_parser.add_argument('--limit', type=int, default=5,
                             help='Results per query (default: 5)')
    batch_parser.add_argument('--output', help='Output file (default: stdout)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        searcher = OptimizedJSONSearcher(args.db_file, args.index_db)
        
        if args.command == 'index':
            print(f"Building index for table '{args.table}'...")
            stats = searcher.build_index(args.table)
            print(f"\n✅ Index built successfully!")
            print(f"   • Documents indexed: {stats['total_documents']:,}")
            print(f"   • JSON columns: {', '.join(stats['json_columns'])}")
            print(f"   • Index size: {stats['index_size_mb']} MB")
            print(f"   • Build time: {stats['index_time']:.2f} seconds")
            if args.index_db != ':memory:':
                print(f"   • Index saved to: {args.index_db}")
        
        elif args.command == 'search':
            # Check if index exists
            try:
                searcher.conn.execute("SELECT 1 FROM fts_data LIMIT 1")
            except:
                print("❌ Index not found. Please run 'index' command first.")
                sys.exit(1)
            
            keywords = args.keywords.split()
            results = searcher.search(keywords, args.limit)
            
            if args.json:
                print(json.dumps(results, indent=2))
            else:
                print(f"\nSearch results for: {args.keywords}")
                print("-" * 60)
                if not results:
                    print("No matching documents found.")
                else:
                    for i, result in enumerate(results, 1):
                        print(f"\n#{i}. Hash: {result['hash']}")
                        print(f"    Score: {result['score']}")
                        print(f"    Snippet: {result['snippet']}...")
        
        elif args.command == 'batch_search':
            # Read keywords file
            with open(args.keywords_file, 'r') as f:
                queries = [line.strip().split() for line in f if line.strip()]
            
            print(f"Processing {len(queries)} queries...")
            results = searcher.batch_search(queries, args.limit)
            
            output = json.dumps(results, indent=2)
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output)
                print(f"✅ Results saved to: {args.output}")
            else:
                print(output)
                
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

python duckdb_json_search_cli.py index mydata.duckdb my_table --index-db ./search.db

# Search for keywords  
python duckdb_json_search_cli.py search mydata.duckdb my_table "keyword1 keyword2"
