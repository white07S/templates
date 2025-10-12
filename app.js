import duckdb
from duckdb_extensions import import_extension

con = duckdb.connect("vectors.duckdb")
import_extension("vss")  # loads the bundled binary offline

# now use VSS as usual
con.sql("CREATE TABLE IF NOT EXISTS docs(id TEXT, emb FLOAT[768]);")
con.sql("CREATE INDEX IF NOT EXISTS idx ON docs USING HNSW(emb) WITH (metric='cosine');")
