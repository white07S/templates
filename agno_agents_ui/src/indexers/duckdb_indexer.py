"""DuckDB indexer for dataframe operations."""

import duckdb
import os
import pyarrow.parquet as pq
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import json


class DuckDBIndexer:
    """Handle DuckDB operations for dataframe-based search."""

    def __init__(self, index_path: str):
        """Initialize DuckDB indexer.

        Args:
            index_path: Path to store the DuckDB database
        """
        self.index_path = index_path
        self.db_path = os.path.join(index_path, "duckdb_index.db")
        self.conn = None

    def connect(self):
        """Establish connection to DuckDB."""
        self.conn = duckdb.connect(self.db_path)
        # Set memory limit and threads for better performance
        self.conn.execute("SET memory_limit='8GB';")
        self.conn.execute("SET threads TO 8;")

    def close(self):
        """Close DuckDB connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def index_exists(self) -> bool:
        """Check if DuckDB index exists."""
        if not os.path.exists(self.db_path):
            return False

        try:
            self.connect()
            result = self.conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'documents'"
            ).fetchone()
            self.close()
            return result[0] > 0
        except:
            return False

    def create_index(self, parquet_path: str, force_reindex: bool = False):
        """Create DuckDB index from parquet file.

        Args:
            parquet_path: Path to the parquet file
            force_reindex: Whether to force reindexing
        """
        if self.index_exists() and not force_reindex:
            print("DuckDB index already exists. Skipping...")
            return

        print("Creating DuckDB index...")
        os.makedirs(self.index_path, exist_ok=True)

        self.connect()

        try:
            # Drop table if exists
            self.conn.execute("DROP TABLE IF EXISTS documents")

            # Create table from parquet file directly
            print("Loading parquet file into DuckDB...")
            self.conn.execute(f"""
                CREATE TABLE documents AS
                SELECT * FROM read_parquet('{parquet_path}')
            """)

            # Create indexes on commonly searched columns
            print("Creating column indexes...")
            indexed_columns = ['id', 'category', 'status', 'priority', 'type',
                              'department', 'country', 'language', 'is_active']

            for col in tqdm(indexed_columns, desc="Creating indexes"):
                try:
                    self.conn.execute(f"CREATE INDEX idx_{col} ON documents({col})")
                except Exception as e:
                    print(f"Warning: Could not create index for {col}: {e}")

            # Get row count
            row_count = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            print(f"✓ DuckDB index created with {row_count:,} rows")

        finally:
            self.close()

    def search(self, column_value_pairs: Dict[str, Any], limit: Optional[int] = None) -> tuple[List[Dict], int]:
        """Search documents using column-value pairs.

        Args:
            column_value_pairs: Dictionary of column-value pairs to filter
            limit: Optional limit on results

        Returns:
            Tuple of (results list, total count)
        """
        self.connect()

        try:
            # Build WHERE clause
            where_conditions = []
            params = []

            for col, val in column_value_pairs.items():
                if val is None:
                    where_conditions.append(f"{col} IS NULL")
                elif isinstance(val, str):
                    where_conditions.append(f"{col} = ?")
                    params.append(val)
                elif isinstance(val, (list, tuple)):
                    placeholders = ','.join(['?'] * len(val))
                    where_conditions.append(f"{col} IN ({placeholders})")
                    params.extend(val)
                else:
                    where_conditions.append(f"{col} = ?")
                    params.append(val)

            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"

            # Get total count
            count_query = f"SELECT COUNT(*) FROM documents WHERE {where_clause}"
            total_count = self.conn.execute(count_query, params).fetchone()[0]

            # Get results
            if limit:
                query = f"SELECT * FROM documents WHERE {where_clause} LIMIT {limit}"
            else:
                query = f"SELECT * FROM documents WHERE {where_clause}"

            result = self.conn.execute(query, params).fetchdf()

            # Convert to list of dicts
            results = result.to_dict('records') if not result.empty else []

            # Convert datetime objects to strings for JSON serialization
            for row in results:
                for key, value in row.items():
                    if hasattr(value, 'isoformat'):
                        row[key] = value.isoformat()

            return results, total_count

        finally:
            self.close()

    def get_column_names(self) -> List[str]:
        """Get list of column names in the database."""
        self.connect()
        try:
            result = self.conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'documents'"
            ).fetchall()
            return [row[0] for row in result]
        finally:
            self.close()

    def get_row_count(self) -> int:
        """Get total number of rows in the database."""
        self.connect()
        try:
            return self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        finally:
            self.close()