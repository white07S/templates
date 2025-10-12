import duckdb, numpy as np
import os

# connect to file-based DB
con = duckdb.connect("vectors.duckdb")

# manually load local extension
ext_path = os.path.abspath("ext/vss.duckdb_extension")
con.load_extension(ext_path)

# confirm
print(con.execute("SELECT * FROM duckdb_extensions()").fetchall())

# create schema
con.execute("""
CREATE TABLE IF NOT EXISTS docs (
  id TEXT PRIMARY KEY,
  text TEXT,
  emb FLOAT[768]
);
""")

# insert sample data
ids = [f"d{i}" for i in range(3)]
txt = ["hello world", "risk modeling at UBS", "badminton tips"]
vecs = np.random.rand(3, 768).astype("float32")
con.executemany("INSERT INTO docs VALUES (?, ?, ?)", zip(ids, txt, vecs))

# build ANN index
con.execute("""
CREATE INDEX IF NOT EXISTS docs_emb_hnsw ON docs USING HNSW(emb)
WITH (metric='cosine');
""")

# sample search
q = np.random.rand(768).astype("float32")
res = con.execute("""
SELECT id, text, vss_distance(emb, ?) AS score
FROM docs
USING SAMPLE BY HNSW(emb) WITH (metric='cosine')
ORDER BY score
LIMIT 5;
""", [q]).fetchall()

print(res)
