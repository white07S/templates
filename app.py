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
    
    def __init__(self, db_path: str, index_path: Optional[str] = None):
        """
        Initialize the search system.
        
        Args:
            db_path: Path to the DuckDB file to search
            index_path: Optional path for index database (default: in-memory)
        """
        self.db_path = db_path
        self.index_path = index_path or ':memory:'
        self.conn = None
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
        """Establish connection and configure for performance."""
        self.conn = duckdb.connect(self.index_path)
        
        # Configure for performance
        self.conn.execute("""
            SET memory_limit = '4GB';
            SET threads = 4;
            INSTALL fts;
            LOAD fts;
        """)
        
        # Attach source database
        self.conn.execute(f"ATTACH '{self.db_path}' AS source_db (READ_ONLY)")
    
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
        schema = self.conn.execute(f"""
            SELECT column_name, data_type 
            FROM source_db.information_schema.columns 
            WHERE table_name = '{table_name}'
        """).fetchall()
        
        if not schema:
            raise ValueError(f"Table '{table_name}' not found in database")
        
        columns = {}
        for col_name, col_type in schema:
            columns[col_name] = col_type
            # Identify JSON columns (could be JSON or VARCHAR containing JSON)
            if col_type in ['JSON', 'VARCHAR'] and col_name != 'hash':
                self.json_columns.append(col_name)
        
        # Verify hash column exists
        if 'hash' not in columns:
            raise ValueError("Table must have a 'hash' column as primary key")
            
        if not self.json_columns:
            raise ValueError("No JSON columns found in table")
            
        return columns
    
    def build_index(self, table_name: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Build the full-text search index.
        
        Args:
            table_name: Name of the table to index
            verbose: Whether to print progress messages
            
        Returns:
            Dictionary with indexing statistics
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"DuckDB JSON Full-Text Search - Index Building")
            print(f"{'='*60}")
        
        start_time = time.time()
        stats = {}
        
        # Discover schema
        columns = self.discover_schema(table_name)
        stats['json_columns'] = self.json_columns
        
        if verbose:
            print(f"✓ Found table '{table_name}' with columns: {list(columns.keys())}")
            print(f"✓ JSON columns identified: {self.json_columns}")
        
        # Build extraction query using json_each for better performance
        if verbose:
            print("\n⏳ Extracting JSON values...")
        
        extract_start = time.time()
        extractions = []
        
        for col in self.json_columns:
            extractions.append(f"""
                SELECT 
                    t.hash,
                    '{col}' || '.' || e.key || ':' || e.value as text_fragment
                FROM source_db.{table_name} t,
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
        
        # Drop existing tables if any
        self.conn.execute("""
            DROP TABLE IF EXISTS fts_data;
            DROP SCHEMA IF EXISTS fts_main_fts_data CASCADE;
        """)
        
        self.conn.execute(f"""
            CREATE TABLE fts_data AS
            SELECT 
                hash,
                STRING_AGG(text_fragment, ' ') as content,
                -- Store metadata for result enrichment
                JSON_OBJECT(
                    'fragment_count', COUNT(*),
                    'columns', LIST(DISTINCT SPLIT_PART(text_fragment, '.', 1))
                ) as metadata
            FROM ({union_query}) fragments
            GROUP BY hash
        """)
        
        stats['extraction_time'] = time.time() - extract_start
        stats['total_documents'] = self.conn.execute(
            "SELECT COUNT(*) FROM fts_data"
        ).fetchone()[0]
        
        if verbose:
            print(f"✓ Extracted values in {stats['extraction_time']:.2f}s")
            print(f"✓ Documents prepared: {stats['total_documents']:,}")
        
        # Create FTS index
        if verbose:
            print("\n⏳ Building FTS index...")
        
        index_start = time.time()
        
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
        
        stats['index_time'] = time.time() - index_start
        self.index_created = True
        
        # Get index statistics
        index_stats = self.conn.execute("""
            SELECT 
                COUNT(DISTINCT term) as unique_terms,
                COUNT(*) as total_terms
            FROM fts_main_fts_data.terms
        """).fetchone()
        
        stats['unique_terms'] = index_stats[0]
        stats['total_terms'] = index_stats[1]
        stats['total_time'] = time.time() - start_time
        
        # Save metadata
        self._save_metadata(table_name, stats)
        
        if verbose:
            print(f"✓ FTS index created in {stats['index_time']:.2f}s")
            print(f"✓ Unique terms indexed: {stats['unique_terms']:,}")
            print(f"\n✅ Total indexing time: {stats['total_time']:.2f}s")
            if self.index_path != ':memory:':
                print(f"✅ Index saved to: {self.index_path}")
            print(f"{'='*60}\n")
        
        return stats
    
    def search(self, keywords: List[str], limit: int = 5, 
               return_metadata: bool = False) -> List[Dict[str, Any]]:
        """
        Search for documents matching the given keywords.
        
        Args:
            keywords: List of keywords to search
            limit: Number of top results to return
            return_metadata: Whether to include metadata in results
            
        Returns:
            List of matching documents with scores
        """
        if not self.index_created:
            # Try to verify if index exists
            try:
                self.conn.execute("SELECT 1 FROM fts_data LIMIT 1").fetchone()
                self.index_created = True
            except:
                raise RuntimeError("Index not found. Call build_index() first.")
        
        # Prepare search query
        search_query = ' '.join(keywords)
        
        # Perform BM25 search
        query = f"""
            SELECT 
                hash,
                ROUND(fts_main_fts_data.match_bm25(
                    hash, 
                    ?,
                    k = 1.2,
                    b = 0.75
                ), 4) as score,
                SUBSTR(content, 1, 200) as snippet
                {', metadata' if return_metadata else ''}
            FROM fts_data
            WHERE score IS NOT NULL
            ORDER BY score DESC
            LIMIT ?
        """
        
        results = self.conn.execute(query, [search_query, limit]).fetchall()
        
        # Format results
        formatted_results = []
        for row in results:
            result = {
                'hash': row[0],
                'score': float(row[1]),
                'snippet': row[2][:150] + '...' if len(row[2]) > 150 else row[2]
            }
            
            if return_metadata and len(row) > 3:
                metadata = json.loads(row[3]) if row[3] else {}
                result['matching_columns'] = metadata.get('columns', [])
                result['fragment_count'] = metadata.get('fragment_count', 0)
            
            formatted_results.append(result)
        
        return formatted_results
    
    def batch_search(self, queries: List[List[str]], limit: int = 5) -> Dict[str, List]:
        """
        Batch search for multiple queries efficiently.
        
        Args:
            queries: List of keyword lists to search
            limit: Number of results per query
            
        Returns:
            Dictionary mapping query strings to results
        """
        results = {}
        for query_keywords in queries:
            query_key = ' '.join(query_keywords)
            results[query_key] = self.search(query_keywords, limit)
        return results
    
    def get_full_document(self, hash_value: str) -> Dict[str, Any]:
        """
        Retrieve the full document for a given hash.
        
        Args:
            hash_value: The hash ID of the document
            
        Returns:
            Full document with all JSON columns
        """
        if not self.json_columns or not self.table_name:
            raise RuntimeError("Schema not discovered. Call build_index() first.")
        
        columns_str = ', '.join(self.json_columns)
        result = self.conn.execute(f"""
            SELECT hash, {columns_str}
            FROM source_db.{self.table_name}
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
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        if not self.index_created:
            return {}
        
        stats = {}
        
        # Document count
        stats['document_count'] = self.conn.execute(
            "SELECT COUNT(*) FROM fts_data"
        ).fetchone()[0]
        
        # Term statistics
        term_stats = self.conn.execute("""
            SELECT 
                COUNT(DISTINCT term) as unique_terms,
                COUNT(*) as total_terms,
                MAX(df) as max_doc_frequency,
                AVG(df) as avg_doc_frequency
            FROM fts_main_fts_data.terms
        """).fetchone()
        
        stats['unique_terms'] = term_stats[0]
        stats['total_terms'] = term_stats[1]
        stats['max_doc_frequency'] = term_stats[2]
        stats['avg_doc_frequency'] = round(term_stats[3], 2) if term_stats[3] else 0
        
        # Top terms
        stats['top_terms'] = self.conn.execute("""
            SELECT term, df 
            FROM fts_main_fts_data.terms
            ORDER BY df DESC
            LIMIT 10
        """).fetchall()
        
        return stats
    
    def _save_metadata(self, table_name: str, stats: Dict[str, Any]):
        """Save index metadata for later reference."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS index_metadata (
                table_name VARCHAR,
                db_path VARCHAR,
                json_columns JSON,
                doc_count INTEGER,
                unique_terms INTEGER,
                index_time FLOAT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        self.conn.execute("""
            DELETE FROM index_metadata WHERE table_name = ? AND db_path = ?
        """, [table_name, self.db_path])
        
        self.conn.execute("""
            INSERT INTO index_metadata 
            (table_name, db_path, json_columns, doc_count, unique_terms, index_time)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            table_name, 
            self.db_path, 
            json.dumps(self.json_columns),
            stats.get('total_documents', 0),
            stats.get('unique_terms', 0),
            stats.get('total_time', 0)
        ])
    
    def load_existing_index(self) -> bool:
        """
        Try to load an existing index from the index database.
        
        Returns:
            True if index was loaded successfully, False otherwise
        """
        try:
            # Check if index tables exist
            self.conn.execute("SELECT 1 FROM fts_data LIMIT 1").fetchone()
            
            # Load metadata
            metadata = self.conn.execute("""
                SELECT table_name, json_columns 
                FROM index_metadata 
                ORDER BY created_at DESC 
                LIMIT 1
            """).fetchone()
            
            if metadata:
                self.table_name = metadata[0]
                self.json_columns = json.loads(metadata[1])
                self.index_created = True
                return True
        except:
            pass
        
        return False
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    sample_db = "sample_data.duckdb"
    
    print("Creating sample database...")
    with duckdb.connect(sample_db) as conn:
        conn.execute("""
            CREATE OR REPLACE TABLE products (
                hash VARCHAR PRIMARY KEY,
                product_info JSON,
                specifications JSON,
                metadata JSON
            )
        """)
        
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
    
    print(f"✓ Sample database created: {sample_db}\n")
    
    # Use the search system
    with DuckDBJSONSearch(sample_db) as searcher:
        # Build index
        searcher.build_index('products')
        
        # Example searches
        test_searches = [
            ['laptop', 'portable'],
            ['TechCorp'],
            ['gaming', 'rgb'],
            ['wireless', 'bluetooth']
        ]
        
        print("\n" + "="*60)
        print("Search Results")
        print("="*60)
        
        for keywords in test_searches:
            print(f"\n🔍 Searching for: {keywords}")
            print("-" * 40)
            
            results = searcher.search(keywords, limit=3, return_metadata=True)
            
            if not results:
                print("No matching documents found.")
            else:
                for i, result in enumerate(results, 1):
                    print(f"\n#{i}. Hash: {result['hash']}")
                    print(f"    Score: {result['score']}")
                    if 'matching_columns' in result:
                        print(f"    Columns: {', '.join(result['matching_columns'])}")
                    print(f"    Snippet: {result['snippet']}")
        
        # Show index statistics
        print("\n" + "="*60)
        print("Index Statistics")
        print("="*60)
        stats = searcher.get_index_stats()
        print(f"Documents: {stats['document_count']}")
        print(f"Unique terms: {stats['unique_terms']}")
        print(f"Top terms: {[t[0] for t in stats['top_terms'][:5]]}")


#!/usr/bin/env python3
"""
DuckDB JSON Full-Text Search CLI
Command-line interface for the DuckDB JSON search system.

Usage:
    python duckdb_json_fts_cli.py index <db_file> <table_name> [--index-db <path>]
    python duckdb_json_fts_cli.py search <db_file> <table_name> <keywords> [--index-db <path>]
    python duckdb_json_fts_cli.py batch-search <db_file> <table_name> <keywords_file> [--index-db <path>]
    python duckdb_json_fts_cli.py stats [--index-db <path>]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

# Import the main search class
try:
    from duckdb_json_fts import DuckDBJSONSearch
except ImportError:
    # If running as a script, try to import from the same directory
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from duckdb_json_fts import DuckDBJSONSearch


def format_results_table(results: List[dict], verbose: bool = False) -> str:
    """Format search results as a nice table."""
    if not results:
        return "No matching documents found."
    
    output = []
    for i, result in enumerate(results, 1):
        output.append(f"\n#{i}. Hash: {result['hash']}")
        output.append(f"    Score: {result['score']}")
        
        if verbose and 'matching_columns' in result:
            output.append(f"    Columns: {', '.join(result['matching_columns'])}")
        
        # Truncate snippet for display
        snippet = result.get('snippet', '')
        if len(snippet) > 150:
            snippet = snippet[:147] + '...'
        output.append(f"    Snippet: {snippet}")
    
    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(
        description='DuckDB JSON Full-Text Search CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index for a table
  %(prog)s index mydata.duckdb products --index-db ./search_index.duckdb
  
  # Search for keywords (in-memory index)
  %(prog)s search mydata.duckdb products "laptop wireless"
  
  # Search using existing index
  %(prog)s search mydata.duckdb products "gaming rgb" --index-db ./search_index.duckdb
  
  # Batch search from file
  %(prog)s batch-search mydata.duckdb products keywords.txt --index-db ./search_index.duckdb
  
  # View index statistics
  %(prog)s stats --index-db ./search_index.duckdb
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Build search index')
    index_parser.add_argument('db_file', help='DuckDB database file')
    index_parser.add_argument('table', help='Table name to index')
    index_parser.add_argument('--index-db', 
                             help='Index database path (default: in-memory)')
    index_parser.add_argument('--quiet', action='store_true',
                             help='Suppress progress messages')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for keywords')
    search_parser.add_argument('db_file', help='DuckDB database file')
    search_parser.add_argument('table', help='Table name')
    search_parser.add_argument('keywords', help='Keywords to search (space-separated)')
    search_parser.add_argument('--index-db',
                              help='Index database path (default: in-memory)')
    search_parser.add_argument('--limit', type=int, default=5,
                              help='Number of results (default: 5)')
    search_parser.add_argument('--json', action='store_true',
                              help='Output as JSON')
    search_parser.add_argument('--verbose', action='store_true',
                              help='Show additional metadata')
    search_parser.add_argument('--rebuild', action='store_true',
                              help='Rebuild index before searching')
    
    # Batch search command
    batch_parser = subparsers.add_parser('batch-search', 
                                         help='Batch search from file')
    batch_parser.add_argument('db_file', help='DuckDB database file')
    batch_parser.add_argument('table', help='Table name')
    batch_parser.add_argument('keywords_file', 
                             help='File with keywords (one query per line)')
    batch_parser.add_argument('--index-db',
                             help='Index database path (default: in-memory)')
    batch_parser.add_argument('--limit', type=int, default=5,
                             help='Results per query (default: 5)')
    batch_parser.add_argument('--output', help='Output file (default: stdout)')
    batch_parser.add_argument('--rebuild', action='store_true',
                             help='Rebuild index before searching')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', 
                                         help='View index statistics')
    stats_parser.add_argument('--index-db', required=True,
                              help='Index database path')
    stats_parser.add_argument('--json', action='store_true',
                              help='Output as JSON')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'index':
            # Build index
            print(f"Building index for table '{args.table}'...")
            
            with DuckDBJSONSearch(args.db_file, args.index_db) as searcher:
                stats = searcher.build_index(args.table, verbose=not args.quiet)
                
                if args.quiet:
                    print(f"✅ Index built successfully!")
                    print(f"   • Documents: {stats['total_documents']:,}")
                    print(f"   • Terms: {stats['unique_terms']:,}")
                    print(f"   • Time: {stats['total_time']:.2f}s")
                    if args.index_db:
                        print(f"   • Saved to: {args.index_db}")
        
        elif args.command == 'search':
            # Search
            with DuckDBJSONSearch(args.db_file, args.index_db) as searcher:
                # Build or load index
                if args.rebuild:
                    print("Rebuilding index...")
                    searcher.build_index(args.table, verbose=False)
                else:
                    # Try to load existing index
                    if not searcher.load_existing_index():
                        print("No existing index found. Building new index...")
                        searcher.build_index(args.table, verbose=False)
                    else:
                        print(f"Using existing index for table '{searcher.table_name}'")
                
                # Perform search
                keywords = args.keywords.split()
                start_time = time.time()
                results = searcher.search(keywords, args.limit, 
                                        return_metadata=args.verbose)
                search_time = (time.time() - start_time) * 1000
                
                # Output results
                if args.json:
                    output = {
                        'query': args.keywords,
                        'time_ms': round(search_time, 2),
                        'results': results
                    }
                    print(json.dumps(output, indent=2))
                else:
                    print(f"\n🔍 Search results for: {args.keywords}")
                    print(f"   (completed in {search_time:.2f}ms)")
                    print("-" * 60)
                    print(format_results_table(results, args.verbose))
        
        elif args.command == 'batch_search':
            # Batch search
            # Read keywords file
            keywords_file = Path(args.keywords_file)
            if not keywords_file.exists():
                raise FileNotFoundError(f"Keywords file not found: {args.keywords_file}")
            
            with open(keywords_file, 'r') as f:
                queries = [line.strip().split() for line in f if line.strip()]
            
            if not queries:
                raise ValueError("Keywords file is empty")
            
            print(f"Processing {len(queries)} queries...")
            
            with DuckDBJSONSearch(args.db_file, args.index_db) as searcher:
                # Build or load index
                if args.rebuild:
                    print("Rebuilding index...")
                    searcher.build_index(args.table, verbose=False)
                else:
                    if not searcher.load_existing_index():
                        print("Building index...")
                        searcher.build_index(args.table, verbose=False)
                
                # Perform batch search
                start_time = time.time()
                results = searcher.batch_search(queries, args.limit)
                total_time = time.time() - start_time
                
                # Prepare output
                output = {
                    'total_queries': len(queries),
                    'total_time_seconds': round(total_time, 2),
                    'avg_time_ms': round((total_time / len(queries)) * 1000, 2),
                    'results': results
                }
                
                output_json = json.dumps(output, indent=2)
                
                if args.output:
                    with open(args.output, 'w') as f:
                        f.write(output_json)
                    print(f"✅ Results saved to: {args.output}")
                    print(f"   • Queries: {len(queries)}")
                    print(f"   • Total time: {total_time:.2f}s")
                    print(f"   • Avg time: {output['avg_time_ms']}ms per query")
                else:
                    print(output_json)
        
        elif args.command == 'stats':
            # View statistics
            with DuckDBJSONSearch('dummy.db', args.index_db) as searcher:
                if not searcher.load_existing_index():
                    print("❌ No index found at specified path")
                    sys.exit(1)
                
                stats = searcher.get_index_stats()
                
                if args.json:
                    # Convert top_terms tuples to dict for JSON
                    if 'top_terms' in stats:
                        stats['top_terms'] = [
                            {'term': t[0], 'frequency': t[1]} 
                            for t in stats['top_terms']
                        ]
                    print(json.dumps(stats, indent=2))
                else:
                    print("\n📊 Index Statistics")
                    print("=" * 60)
                    print(f"Documents: {stats.get('document_count', 0):,}")
                    print(f"Unique terms: {stats.get('unique_terms', 0):,}")
                    print(f"Total terms: {stats.get('total_terms', 0):,}")
                    print(f"Avg doc frequency: {stats.get('avg_doc_frequency', 0)}")
                    
                    if stats.get('top_terms'):
                        print(f"\nTop 10 terms:")
                        for term, freq in stats['top_terms']:
                            print(f"  • {term}: {freq} documents")
                
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        if '--debug' in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


# Build index (saved to disk for reuse)
python duckdb_json_search_cli.py index data.duckdb my_table \
    --index-db ./search_index.duckdb

# Search for keywords
python duckdb_json_search_cli.py search data.duckdb my_table \
    "keyword1 keyword2" --index-db ./search_index.duckdb

# Batch search from file
python duckdb_json_search_cli.py batch-search data.duckdb my_table \
    keywords.txt --index-db ./search_index.duckdb --output results.json
