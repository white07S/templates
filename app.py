# exact_cosine_top5_torch_adaptive.py
# Pure PyTorch, exact cosine similarity, multi-GPU sharded, adaptive chunking per GPU.
# Input: DataFrame with columns: 'hash' (str) and 'feature_1' (4096-d embedding as list/ndarray)
# Output: prints top-5 (hash, score) for a single 4096-d query vector.

import os
import math
import heapq
import numpy as np
import pandas as pd
import torch


# ------------------------
# Utilities
# ------------------------
def _normalize_rows_f32(mat: np.ndarray) -> np.ndarray:
    """L2-normalize rows in-place (float32)."""
    mat = np.asarray(mat, dtype=np.float32, order="C")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    mat /= norms
    return mat


def build_matrix_from_df(
    df: pd.DataFrame,
    col_hash: str = "hash",
    col_vec: str = "feature_1",
    d_expected: int = 4096,
):
    """
    Convert df[col_vec] (lists/arrays) -> dense float32 matrix [N, d_expected], and normalize rows.
    Returns (X, hashes)
    """
    hashes = df[col_hash].astype(str).tolist()
    N = len(df)
    X = np.empty((N, d_expected), dtype=np.float32)
    for i, v in enumerate(df[col_vec].tolist()):
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] != d_expected:
            raise ValueError(f"Row {i} has dim {arr.shape}, expected ({d_expected},)")
        X[i] = arr
    X = _normalize_rows_f32(X)
    return X, hashes


def _bytes_per_row(d: int, dtype: torch.dtype = torch.float32, safety_overhead: float = 1.2) -> int:
    """
    Approx VRAM per row for GEMV (X_chunk @ q):
      - X_chunk rows: d * bytes_per_el
      - sims output: ~4 bytes per row
      - workspace/overhead ~20%
    """
    bytes_el = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}[dtype]
    per_row = d * bytes_el + 4
    return int(per_row * safety_overhead)


def suggest_rows_per_chunk_vram(
    rows_in_shard: int,
    d: int,
    dtype: torch.dtype = torch.float32,
    device: int = 0,
    vram_safety_frac: float = 0.80,
    target_chunks_range=(16, 64),
    hard_min_rows: int = 8_192,
    hard_max_rows: int | None = None,
) -> int:
    """
    Choose rows_per_chunk using real-time free VRAM and shard size so we:
      - avoid OOM,
      - keep ~16–64 chunks per shard for good throughput,
      - respect absolute min/max.
    """
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device=device)  # (free, total)
    usable = int(free_bytes * vram_safety_frac)

    mem_bound = max(1, usable // _bytes_per_row(d, dtype=dtype))

    tgt_min_chunks, tgt_max_chunks = target_chunks_range
    size_lower = max(1, rows_in_shard // tgt_max_chunks)   # larger chunks (fewer total)
    size_upper = max(1, rows_in_shard // tgt_min_chunks)   # smaller chunks (more total)

    rows = max(size_lower, min(mem_bound, size_upper))
    rows = max(rows, hard_min_rows)
    if hard_max_rows is not None:
        rows = min(rows, hard_max_rows)
    rows = min(rows, rows_in_shard)
    return int(rows)


# ------------------------
# Core search
# ------------------------
@torch.inference_mode()
def exact_cosine_topk_torch(
    hashes: list[str],
    X_np: np.ndarray,          # [N, d] float32, unit-normalized rows
    q_np: np.ndarray,          # [d] float32 (will normalize)
    k: int = 5,
    compute_dtype: torch.dtype = torch.float32,  # try torch.bfloat16 on A100/H100 for speed
    target_chunks_range=(16, 64),
    vram_safety_frac: float = 0.80,
    cpu_chunk_rows: int = 100_000,               # CPU fallback chunk size
):
    """
    Exact cosine similarity (dot product on unit vectors). Multi-GPU sharded.
    Returns top-k [(hash, score)].
    """
    d = X_np.shape[1]
    q = np.asarray(q_np, dtype=np.float32).reshape(d)
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-12:
        raise ValueError("Query vector has near-zero norm.")
    q = q / q_norm

    device_count = torch.cuda.device_count()

    if device_count == 0:
        # -------------------- CPU path (exact) --------------------
        X_t = torch.from_numpy(X_np)  # [N, d]
        q_t = torch.from_numpy(q)
        N = X_t.shape[0]
        heap: list[tuple[float, int]] = []  # (score, global_idx)

        for s in range(0, N, cpu_chunk_rows):
            e = min(s + cpu_chunk_rows, N)
            sims = torch.mv(X_t[s:e], q_t)  # [e-s]
            tk = min(k, e - s)
            vals, idxs = torch.topk(sims, k=tk)
            vals = vals.tolist()
            idxs = idxs.tolist()
            for val, idx in zip(vals, idxs):
                gi = s + idx
                if len(heap) < k:
                    heapq.heappush(heap, (val, gi))
                else:
                    if val > heap[0][0]:
                        heapq.heapreplace(heap, (val, gi))

        top = sorted(heap, key=lambda x: x[0], reverse=True)
        return [(hashes[i], float(s)) for s, i in top]

    # -------------------- Multi-GPU path (exact) --------------------
    torch.set_float32_matmul_precision('high')  # enable TF32 on Ampere+ where applicable

    N = X_np.shape[0]
    base = N // device_count
    rem = N % device_count
    shard_sizes = [base + (1 if i < rem else 0) for i in range(device_count)]
    offsets = [0]
    for sz in shard_sizes[:-1]:
        offsets.append(offsets[-1] + sz)

    # Place query on each device
    q_per_dev = [torch.from_numpy(q).to(f"cuda:{i}", non_blocking=True).to(compute_dtype)
                 for i in range(device_count)]

    global_heap: list[tuple[float, int]] = []  # (score, global_idx)

    for dev_id, (start, sz) in enumerate(zip(offsets, shard_sizes)):
        if sz == 0:
            continue
        end = start + sz

        rows_per_chunk = suggest_rows_per_chunk_vram(
            rows_in_shard=sz,
            d=d,
            dtype=compute_dtype,
            device=dev_id,
            vram_safety_frac=vram_safety_frac,
            target_chunks_range=target_chunks_range,
            hard_min_rows=8_192,
            hard_max_rows=None,  # set e.g. 250_000 if you want an upper cap
        )

        q_t = q_per_dev[dev_id]

        s = start
        while s < end:
            e = min(s + rows_per_chunk, end)
            X_chunk = torch.from_numpy(X_np[s:e]).to(f"cuda:{dev_id}", non_blocking=True).to(compute_dtype)
            sims = torch.mv(X_chunk, q_t)  # exact cosine (both unit-norm)

            tk = min(k, e - s)
            vals, idxs = torch.topk(sims, k=tk)
            vals = vals.detach().cpu().tolist()
            idxs = idxs.detach().cpu().tolist()

            for val, idx in zip(vals, idxs):
                gi = s + idx
                if len(global_heap) < k:
                    heapq.heappush(global_heap, (val, gi))
                else:
                    if val > global_heap[0][0]:
                        heapq.heapreplace(global_heap, (val, gi))

            del X_chunk, sims
            torch.cuda.empty_cache()

            s = e

    top = sorted(global_heap, key=lambda x: x[0], reverse=True)
    return [(hashes[i], float(s)) for s, i in top]


# ------------------------
# Demo / CLI entry
# ------------------------
if __name__ == "__main__":
    """
    Usage:
      - Replace the DEMO block with your real DataFrame + query vector.
      - Ensure df has:
            df["hash"] -> str ids
            df["feature_1"] -> 4096-d vectors (list/ndarray)
      - Run:  python exact_cosine_top5_torch_adaptive.py
      - Optional env:
            TORCH_CUDA_ARCH_LIST, CUDA_VISIBLE_DEVICES
            COMPUTE_DTYPE = "fp32" | "bf16" | "fp16"
    """
    DEMO = os.environ.get("DEMO", "1") == "1"

    if DEMO:
        N, d = 100_000, 4096
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "hash": [f"H_{i:06d}" for i in range(N)],
            "feature_1": [rng.standard_normal(d).astype(np.float32) for _ in range(N)],
        })
        query = rng.standard_normal(d).astype(np.float32)
    else:
        # Example for real data:
        # df = pd.read_parquet("/path/to/your.parquet")
        # query = np.load("/path/to/query.npy").astype(np.float32)  # shape (4096,)
        raise SystemExit("Load your DataFrame and query, or run with DEMO=1.")

    # Prepare matrix + hashes
    X, hashes = build_matrix_from_df(df, "hash", "feature_1", d_expected=4096)

    # Choose compute dtype
    dtype_env = os.environ.get("COMPUTE_DTYPE", "fp32").lower()
    if dtype_env in ("fp16", "float16", "half"):
        compute_dtype = torch.float16
    elif dtype_env in ("bf16", "bfloat16"):
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float32

    top5 = exact_cosine_topk_torch(
        hashes=hashes,
        X_np=X,
        q_np=query,
        k=5,
        compute_dtype=compute_dtype,
        target_chunks_range=(16, 64),
        vram_safety_frac=0.80,
        cpu_chunk_rows=100_000,
    )

    for h, s in top5:
        print(f"{h}\t{s:.6f}")
