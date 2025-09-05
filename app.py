# qdrant_multifeature_search.py
# Requirements:
#   pip install qdrant-client pandas pyarrow numpy

from __future__ import annotations
import hashlib
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointStruct,
    ScrollRequest,
    SearchParams,
    SearchRequest,
    VectorParams,
    WithPayloadSelector,
    PayloadIncludeSelector,
)

# --------- CONFIG ---------
DIM = 4096
MAX_FEATURES = 5
FEATURE_COLS = [f"Feature {i}" for i in range(1, MAX_FEATURES + 1)]
HASH_COL = "Hash"
# Choose 'euclid' for L2 indexing speed; we'll compute exact L1/L2 in final re-rank anyway
QDRANT_DISTANCE = Distance.EUCLID

# Per-feature collections namespace
COLL_PREFIX = "mfset_f"   # -> mfset_f1 ... mfset_f5

# ANN search knobs (tune to your data size)
HNSW_M = 32                 # graph degree
HNSW_EF_CONSTRUCTION = 128  # build-time
HNSW_EF_SEARCH = 128        # query-time (you can raise to 256/512 for more recall)

# Candidate sizes
CANDIDATES_PER_LIST = 80    # K': per (feature, query vector)
FUSE_POOL_LIMIT = 400       # cap union size before exact re-rank

# RRF fusion
RRF_K = 60

# --------- UTILITIES ---------
def stable_point_id(hash_str: str, feature_idx: int, sub_idx: int) -> int:
    """Create a deterministic 63-bit positive integer ID for Qdrant."""
    key = f"{hash_str}|{feature_idx}|{sub_idx}".encode("utf-8")
    h = hashlib.blake2b(key, digest_size=8).digest()
    val = int.from_bytes(h, "big") & ((1 << 63) - 1)
    return val

def to_list_of_vectors(value) -> List[np.ndarray]:
    """
    Normalize a cell into a list[np.ndarray (4096,)].
    Accepts:
      - np.ndarray of shape (4096,)
      - list/tuple of np.ndarray/list (each 4096,)
      - list/tuple of floats of len 4096
      - strings that can be eval'ed to above (last resort)
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []

    def as_vec(x):
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] != DIM:
            raise ValueError(f"Vector must be 1D of length {DIM}, got {arr.shape}")
        return arr

    if isinstance(value, np.ndarray):
        if value.ndim == 1 and value.shape[0] == DIM:
            return [as_vec(value)]
        else:
            raise ValueError(f"Unexpected ndarray shape {value.shape}")

    if isinstance(value, (list, tuple)):
        # Could be a single vector as list-of-floats or list of vectors
        if len(value) == DIM and all(isinstance(t, (float, int, np.floating)) for t in value):
            return [as_vec(value)]
        # else: assume list of vectors
        out = [as_vec(v) for v in value]
        if len(out) > 5:
            raise ValueError("Feature list cannot exceed 5 vectors")
        return out

    if isinstance(value, str):
        # Last-resort parse
        import ast
        parsed = ast.literal_eval(value)
        return to_list_of_vectors(parsed)

    raise ValueError(f"Unsupported cell type for vector(s): {type(value)}")

def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum())

# --------- CORE CLASS ---------
class MultiFeatureQdrant:
    """
    One Qdrant collection per feature (mfset_f1..mfset_f5).
    Each sub-vector is a separate point with payload:
        { "hash": str, "feature": int, "sub_idx": int }
    """

    def __init__(self, host: str = "localhost", port: int = 6333, api_key: Optional[str] = None):
        self.client = QdrantClient(host=host, port=port, api_key=api_key)

    # ---- Collections ----
    def collection_name(self, feature_idx: int) -> str:
        assert 1 <= feature_idx <= MAX_FEATURES
        return f"{COLL_PREFIX}{feature_idx}"

    def ensure_collections(self, recreate: bool = False):
        """
        Create (or recreate) per-feature collections with HNSW index and tuned params.
        """
        for j in range(1, MAX_FEATURES + 1):
            name = self.collection_name(j)
            if recreate and self.client.exists(name):
                self.client.delete_collection(name)

            if not self.client.exists(name):
                self.client.recreate_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=DIM, distance=QDRANT_DISTANCE, on_disk=False  # keep in RAM for speed
                    ),
                    hnsw_config={"m": HNSW_M, "ef_construct": HNSW_EF_CONSTRUCTION},
                    optimizers_config=OptimizersConfigDiff(
                        indexing_threshold=20000,  # build HNSW early
                        memmap_threshold=0,        # keep segments in RAM
                    ),
                    # You can also define payload schema for speed/validation
                    # (optional but nice hygiene)
                    payload_schema={
                        "hash": PayloadSchemaType.keyword,
                        "feature": PayloadSchemaType.integer,
                        "sub_idx": PayloadSchemaType.integer,
                    },
                )
            # Set default query-time ef (you can also pass per-search)
            self.client.set_collection_params(
                name,
                params={"hnsw_config": {"ef_construct": HNSW_EF_CONSTRUCTION}, "optimizer_config": {}},
            )

    # ---- Ingestion ----
    def upsert_dataframe(self, df: pd.DataFrame, batch: int = 2048, dedupe: bool = True):
        """
        Ingest/replace data from a DataFrame with columns:
            Hash (str), Feature 1..Feature 5 (each: 4096D vector or list of <=5 vectors)
        If dedupe=True, existing points for a given Hash across all features are removed first.
        """
        required = [HASH_COL] + FEATURE_COLS
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Missing required column: {c}")

        if dedupe:
            unique_hashes = df[HASH_COL].dropna().astype(str).unique().tolist()
            self._delete_hashes(unique_hashes)

        # Build and send per-feature batches
        for feature_idx, feature_col in enumerate(FEATURE_COLS, start=1):
            name = self.collection_name(feature_idx)
            points: List[PointStruct] = []

            for _, row in df.iterrows():
                h = str(row[HASH_COL])
                try:
                    vecs = to_list_of_vectors(row[feature_col])
                except Exception:
                    # If a feature is missing/empty, skip
                    vecs = []
                # Enforce limit
                if len(vecs) > 5:
                    raise ValueError("Feature list cannot exceed 5 vectors")

                for sub_idx, v in enumerate(vecs):
                    pid = stable_point_id(h, feature_idx, sub_idx)
                    points.append(
                        PointStruct(
                            id=pid,
                            vector=v.astype(np.float32).tolist(),
                            payload={"hash": h, "feature": feature_idx, "sub_idx": sub_idx},
                        )
                    )

                if len(points) >= batch:
                    self.client.upsert(name, points=points)
                    points.clear()

            if points:
                self.client.upsert(name, points=points)

    def _delete_hashes(self, hashes: List[str]):
        """
        Delete all sub-vectors across all feature collections for these hashes.
        """
        if not hashes:
            return
        for j in range(1, MAX_FEATURES + 1):
            name = self.collection_name(j)
            if not self.client.exists(name):
                continue
            flt = Filter(
                must=[FieldCondition(key="hash", match=MatchAny(any=hashes))]
            )
            self.client.delete(name, points_selector=flt)

    # ---- Search ----
    def search(
        self,
        query: Dict[int, Iterable[np.ndarray]],   # {feature_idx: list of vectors} (vectors can be 1 or many)
        top_k: int = 5,
        per_list_k: int = CANDIDATES_PER_LIST,
        prefer_common: bool = True,
        exact_metric: str = "l2",  # "l2" or "l1"
        hnsw_ef: int = HNSW_EF_SEARCH,
    ) -> List[Tuple[str, float]]:
        """
        Returns top_k [(hash, fused_score)].
        - Stage 1: ANN per (feature, qvec) ==> per_list top-K', de-dup by hash within list
        - Stage 2: RRF + coverage fusion across lists to form a candidate pool
        - Stage 3: Exact re-rank (L1/L2) on candidate pool using all sub-vectors per feature
        """

        # Normalize query to {feature_idx: [np.ndarray, ...]}
        qnorm: Dict[int, List[np.ndarray]] = {}
        for feat, qv in query.items():
            if not (1 <= feat <= MAX_FEATURES):
                raise ValueError(f"feature_idx must be 1..{MAX_FEATURES}, got {feat}")
            qnorm[feat] = [np.asarray(v, dtype=np.float32).reshape(-1) for v in qv]
            for v in qnorm[feat]:
                if v.shape[0] != DIM:
                    raise ValueError(f"Query vector must be 1D of length {DIM}, got {v.shape}")

        # ---- Stage 1: Batch ANN per (feature, qvec)
        lists_by_feat_and_q = {}  # key=(feat, qidx) -> list[(hash, distance, rank)]
        for feat, qvecs in qnorm.items():
            name = self.collection_name(feat)
            reqs = [
                SearchRequest(
                    vector=qv.tolist(),
                    limit=per_list_k,
                    with_payload=WithPayloadSelector(include=PayloadIncludeSelector(fields=["hash", "sub_idx"])),
                    params=SearchParams(hnsw_ef=hnsw_ef, exact=False),
                )
                for qv in qvecs
            ]
            results = self.client.search_batch(name, reqs)
            for qidx, res in enumerate(results):
                # de-dup by hash within this (feat,qidx)
                best: Dict[str, float] = {}
                for r in res:
                    h = r.payload["hash"]
                    # Qdrant returns lower score for euclid (distance). Keep min distance per hash.
                    d = float(r.score)
                    if (h not in best) or (d < best[h]):
                        best[h] = d
                ranked = sorted(best.items(), key=lambda x: x[1])  # asc by distance
                lists_by_feat_and_q[(feat, qidx)] = [(h, dist, rank + 1) for rank, (h, dist) in enumerate(ranked)]

        # ---- Stage 2: RRF + coverage fusion
        fused, candidate_hashes = self._rrf_fuse(lists_by_feat_and_q, prefer_common=prefer_common)

        # Cap candidate pool for re-rank
        candidate_hashes = list(candidate_hashes)[:FUSE_POOL_LIMIT]

        # ---- Stage 3: Exact re-rank (L1/L2) using all sub-vectors of each candidate per feature
        exact_scores = self._exact_rerank(qnorm, candidate_hashes, metric=exact_metric)

        # Combine features via best-over-subvecs, then sum over features (or average).
        # exact_scores: dict[hash][feat] = best_distance_for_that_feat
        combined = []
        for h in exact_scores:
            # Only include features that were actually queried
            dists = [exact_scores[h][f] for f in qnorm.keys() if f in exact_scores[h]]
            if not dists:
                continue
            # Convert to similarity-like score for final sorting (higher better)
            sims = [1.0 / (1.0 + d) for d in dists]
            agg_sim = float(np.mean(sims))  # average similarity across features
            combined.append((h, agg_sim))

        # Prefer common hashes first if requested, break ties by agg_sim
        if prefer_common:
            coverage_map = self._coverage_from_lists(lists_by_feat_and_q)
            combined.sort(key=lambda x: (coverage_map.get(x[0], 1), x[1]), reverse=True)
        else:
            combined.sort(key=lambda x: x[1], reverse=True)

        return combined[:top_k]

    # --- helpers: fusion, coverage, re-rank ---
    def _rrf_fuse(self, lists_by_feat_and_q, prefer_common=True):
        # RRF aggregation
        rrf = {}
        coverage = {}
        for (feat, qidx), ranked in lists_by_feat_and_q.items():
            seen = set()
            for (h, _dist, r) in ranked:
                if h in seen:
                    continue
                seen.add(h)
                rrf[h] = rrf.get(h, 0.0) + 1.0 / (RRF_K + r)
                coverage.setdefault(h, set()).add(feat)

        items = [(h, rrf[h], len(coverage[h])) for h in rrf.keys()]
        if prefer_common:
            common = [x for x in items if x[2] >= 2]
            pool = common if len(common) >= 5 else items
        else:
            pool = items

        pool.sort(key=lambda x: (x[1], x[2], x[0]), reverse=True)
        # candidate set by decreasing fused score
        candidate_hashes = [h for (h, _, _) in pool]
        return pool, candidate_hashes

    def _coverage_from_lists(self, lists_by_feat_and_q):
        coverage = {}
        for (feat, _qidx), ranked in lists_by_feat_and_q.items():
            for (h, _d, _r) in ranked:
                coverage.setdefault(h, set()).add(feat)
        return {h: len(feats) for h, feats in coverage.items()}

    def _exact_rerank(
        self,
        qnorm: Dict[int, List[np.ndarray]],
        candidate_hashes: List[str],
        metric: str = "l2",
    ) -> Dict[str, Dict[int, float]]:
        """
        For each candidate hash and each queried feature, compute the BEST distance across:
            all (query vectors) x (sub-vectors of candidate for that feature).
        Returns: dict[hash][feature] = best_distance
        """
        dist_fn = l2 if metric.lower() == "l2" else l1

        # Fetch all sub-vectors for candidates, batched per feature
        per_feature_vectors: Dict[int, Dict[str, List[np.ndarray]]] = {f: {} for f in qnorm.keys()}

        for feat in qnorm.keys():
            name = self.collection_name(feat)
            # Scroll with hash in candidates
            remaining = set(candidate_hashes)
            page = None
            while remaining:
                batch = list(list(remaining)[:256])  # reasonable filter size
                remaining -= set(batch)
                flt = Filter(must=[FieldCondition(key="hash", match=MatchAny(any=batch))])
                points, next_page = self.client.scroll(
                    name,
                    scroll_filter=flt,
                    with_vectors=True,
                    with_payload=WithPayloadSelector(include=PayloadIncludeSelector(fields=["hash", "sub_idx"])),
                    limit=2048,
                    offset=page,
                )
                for p in points:
                    h = p.payload["hash"]
                    per_feature_vectors[feat].setdefault(h, []).append(np.asarray(p.vector, dtype=np.float32))
                if not next_page:
                    continue

        # Compute best distances
        out: Dict[str, Dict[int, float]] = {}
        for feat, qvecs in qnorm.items():
            cand_map = per_feature_vectors.get(feat, {})
            for h, subvecs in cand_map.items():
                best = float("inf")
                # best over (q in qvecs) x (v in subvecs)
                for q in qvecs:
                    # small optimization: vectorize over subvecs
                    M = np.stack(subvecs, axis=0)  # (n, 4096)
                    if dist_fn is l2:
                        # ||M - q||_2
                        dists = np.linalg.norm(M - q[None, :], axis=1)
                    else:
                        dists = np.abs(M - q[None, :]).sum(axis=1)
                    d = float(dists.min())
                    if d < best:
                        best = d
                out.setdefault(h, {})[feat] = best
        return out


# --------- EXAMPLE USAGE ---------
if __name__ == "__main__":
    """
    Example:
      1) Initialize & (re)create collections
      2) Ingest a sample parquet file
      3) Run searches with single-vector and list-of-vectors queries
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, required=False, help="Path to input parquet with Hash + Feature 1..5")
    parser.add_argument("--host", type=str, default=os.getenv("QDRANT_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")))
    parser.add_argument("--api_key", type=str, default=os.getenv("QDRANT_API_KEY"))
    parser.add_argument("--recreate", action="store_true", help="Drop & recreate collections")
    args = parser.parse_args()

    engine = MultiFeatureQdrant(host=args.host, port=args.port, api_key=args.api_key)
    engine.ensure_collections(recreate=args.recreate)

    if args.parquet:
        df = pd.read_parquet(args.parquet)
        engine.upsert_dataframe(df, batch=2048, dedupe=True)
        print("Ingestion done.")

    # ---- Demo search payloads (replace with your real queries)
    # Single-vector query for Feature 1
    q1 = {1: [np.random.randn(DIM).astype(np.float32)]}
    res1 = engine.search(q1, top_k=5, per_list_k=80, prefer_common=True, exact_metric="l2", hnsw_ef=HNSW_EF_SEARCH)
    print("\nSingle-feature query (F1) -> Top-5 hashes:")
    for h, s in res1:
        print(f"{h}\t score={s:.6f}")

    # Multi-feature, multi-vector query (F1 + F3)
    q2 = {
        1: [np.random.randn(DIM).astype(np.float32), np.random.randn(DIM).astype(np.float32)],
        3: [np.random.randn(DIM).astype(np.float32)]
    }
    res2 = engine.search(q2, top_k=5, per_list_k=100, prefer_common=True, exact_metric="l1", hnsw_ef=256)
    print("\nMulti-feature query (F1 & F3, multi-vectors) -> Top-5 hashes:")
    for h, s in res2:
        print(f"{h}\t score={s:.6f}")
