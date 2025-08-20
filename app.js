Got you. Here’s a tight, working check + how to pin Milvus data to your own folder so you can reinstall and remount later.

⸻

1) Quick Python check (connection + tiny round-trip)

# pip install "pymilvus>=2.4"
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

MILVUS_URI = "http://127.0.0.1:19530"   # adjust if remote or TLS
ALIAS = "default"

# 1) connect
connections.connect(alias=ALIAS, uri=MILVUS_URI)

# 2) basic health: list collections (will be empty on fresh install)
print("Existing collections:", utility.list_collections())

# 3) create a small test collection
coll_name = "conn_smoke_test"
if utility.has_collection(coll_name):
    utility.drop_collection(coll_name)

schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=16),
    ],
    description="connectivity smoke test"
)
c = Collection(name=coll_name, schema=schema)

# 4) insert tiny batch
c.insert([
    [0, 1, 2],
    [[0,0,0,0], [0.1,0.1,0.1,0.1], [0.9,0.9,0.9,0.9]],
    ["a", "b", "c"]
])

# 5) build simple index & load
c.create_index("vec", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 16}})
c.load()

# 6) search
res = c.search(data=[[0.08,0.08,0.08,0.08]], anns_field="vec", param={"nprobe": 8}, limit=3, output_fields=["label"])
print("Top hits:", [hit.id for hit in res[0]], [hit.entity.get("label") for hit in res[0]])

# 7) cleanup
c.release()
utility.drop_collection(coll_name)

print("Milvus connectivity OK.")

If this prints “Milvus connectivity OK.” and shows hits, you’re good.

⸻

2) Persist data to your OWN folder (survives reinstalls)

Milvus uses:
	•	Object storage for segments/index files (MinIO or S3-compatible).
	•	A local disk path for cached/loaded data used during query/search (localStorage.path).
	•	etcd for metadata; its key prefix controls which logical Milvus “universe” you’re in.

To make a reinstall painless, externalize these to a stable location and keep the same values after reinstall.

Edit /etc/milvus/configs/milvus.yaml

# --- Object storage (MinIO or S3-compatible) ---
minio:
  address: 127.0.0.1
  port: 9000
  accessKeyID: "minioadmin"
  secretAccessKey: "minioadmin"
  useSSL: false
  bucketName: "milvus"
  rootPath: "segments"     # optional prefix inside the bucket

# --- Local disk working set / cache (set this to your dedicated disk) ---
localStorage:
  path: /srv/milvus/localstore    # create this on a big, fast disk (NVMe if you have it)

# --- etcd (metadata) ---
etcd:
  endpoints: ["localhost:2379"]
  rootPath: "by-dev"              # keep this constant; changing it makes Milvus look empty
  metaSubPath: "meta"
  kvSubPath: "kv"

	•	localStorage.path is the documented knob to move Milvus’ on-disk working set/cache to your own folder (set it before heavy use; moving later affects access to older data).  ￼
	•	Object storage settings under minio: tell Milvus where to persist segments and indices; put MinIO’s data dir (the server’s storage) on your durable disk too.  ￼
	•	Keeping the same etcd.rootPath lets a fresh Milvus binary discover existing metadata in the same etcd namespace.  ￼

If you use the embedded WAL “Woodpecker”

You can also point its storage to your folder (optional, 2.6+):

mq:
  type: woodpecker

woodpecker:
  storage:
    type: local        # or 'minio' for S3-compatible
    rootPath: /srv/milvus/woodpecker

Woodpecker supports local (filesystem) or minio backends; set the root path to your durable disk.  ￼

⸻

3) Practical “reinstall & remount” recipe
	1.	Put everything on a persistent disk:

	•	MinIO server data dir (e.g., /srv/minio/data)
	•	Milvus localStorage.path (e.g., /srv/milvus/localstore)
	•	If using Woodpecker: /srv/milvus/woodpecker

	2.	Backup your config:

	•	/etc/milvus/configs/milvus.yaml
	•	MinIO creds / bucket name

	3.	After reinstall:

	•	Reinstall/run MinIO pointing at the same /srv/minio/data and same bucket.
	•	Restore the same milvus.yaml with the same localStorage.path and etcd.rootPath.
	•	Start Milvus with:

MILVUSCONF=/etc/milvus/configs /usr/bin/milvus run standalone



This way, Milvus will see the same metadata (etcd), the same object storage (segments/index), and the same on-disk path for local working sets.

⸻

Notes / gotchas
	•	Changing localStorage.path after you’ve used Milvus can affect access to old data; set it once and keep it.  ￼
	•	If you ever switch object storage location/bucket, use the Milvus Backup tool to migrate/restore cleanly.  ￼

If you share your current milvus.yaml (sans secrets), I can tailor the exact block for your layout (single disk vs. separate disks for MinIO/working set).
