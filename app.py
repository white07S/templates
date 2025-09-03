# fts_json_anyvalue.py
import duckdb
from typing import List, Tuple

def build_fts(
    db_path: str,
    table: str,
    id_col: str = "hash",
    schema: str = "main",
    stemmer: str = "none",        # avoid over-stemming domain terms/IDs
    stopwords: str = "none",      # keep everything for maximum recall
    ignore_regex: str = r"([^a-z0-9_-])+",  # keep digits + hyphens/underscores
    lower: bool = True,
    strip_accents: bool = True,
    overwrite: bool = True,
) -> List[str]:
    con = duckdb.connect(db_path)
    # Fast & predictable execution settings (optional)
    con.execute("INSTALL json; LOAD json; INSTALL fts; LOAD fts;")
    con.execute("SET enable_progress_bar = true;")
    # Consider tuning these for your box:
    # con.execute("SET threads = 0");  # default uses all cores
    # con.execute("SET memory_limit = '8GB';")

    # 1) Discover JSON columns (excluding the id)
    json_cols = [r[0] for r in con.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = ? AND table_name = ? AND data_type = 'JSON'
          AND column_name <> ?
        ORDER BY ordinal_position
        """, [schema, table, id_col]).fetchall()
    ]
    if not json_cols:
        raise RuntimeError("No JSON columns found")

    # 2) Flatten ALL JSON columns' leaf values, concat per row
    unions = []
    for c in json_cols:
        unions.append(f"""
            SELECT t.{id_col} AS {id_col},
                   CAST(j.value AS VARCHAR) AS v
            FROM {schema}.{table} t,
                 LATERAL json_tree(t.{c}) AS j
            WHERE j.type NOT IN ('object','array')
              AND j.value IS NOT NULL
              AND length(trim(CAST(j.value AS VARCHAR))) > 0
        """)
    kv_sql = "\nUNION ALL\n".join(unions)

    con.execute(f"""
        CREATE OR REPLACE TABLE {schema}.docs_fts AS
        SELECT {id_col}, string_agg(v, ' ') AS text
        FROM (
          {kv_sql}
        ) s
        GROUP BY {id_col};
    """)

    # 3) Build (or overwrite) the FTS index on the materialized text
    con.execute(f"""
      PRAGMA create_fts_index(
        '{schema}.docs_fts', '{id_col}', 'text',
        stemmer='{stemmer}',
        stopwords='{stopwords}',
        ignore='{ignore_regex}',
        strip_accents={1 if strip_accents else 0},
        lower={1 if lower else 0},
        overwrite={1 if overwrite else 0}
      );
    """)
    return json_cols

def search_any(
    db_path: str,
    query_terms: List[str],
    k: int = 5,
    conjunctive: bool = False,
    schema: str = "main",
    id_col: str = "hash",
) -> List[Tuple[str, float]]:
    con = duckdb.connect(db_path)
    q = " ".join(query_terms)
    # The index created above lives under schema fts_main_docs_fts
    rows = con.execute(f"""
        SELECT d.{id_col}, score
        FROM (
          SELECT {id_col},
                 fts_main_docs_fts.match_bm25(
                   {id_col}, ?, fields := NULL,
                   conjunctive := {1 if conjunctive else 0}
                 ) AS score
          FROM {schema}.docs_fts
        ) d
        WHERE score IS NOT NULL
        ORDER BY score DESC
        LIMIT {k};
    """, [q]).fetchall()
    return rows

if __name__ == "__main__":
    # Example usage:
    db = "your_data.duckdb"
    table = "issues_dataset"      # must have a unique 'hash' column + JSON columns
    build_fts(db, table)          # (re)build index
    print(search_any(db, ["segfault", "control", "ISS-0042"], k=5))
