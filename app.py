# pip install httpx uvloop aiofiles
import asyncio, json, os, time, random, hashlib
import httpx
import aiofiles

# -------- Config --------
VLLM_URL = "http://localhost:8000/v1/chat/completions"  # OpenAI-compatible vLLM endpoint
API_KEY = "EMPTY"  # if your server enforces it; otherwise ignore header
MODEL = "your-model-name"
CONCURRENCY = 64          # tune: 32–128 usually safe; align with server capacity
REQUESTS = 200            # how many to fire
TIMEOUT_S = 60            # per-request timeout
CACHE_DIR = "./cache"     # file-based cache directory
# ------------------------

os.makedirs(CACHE_DIR, exist_ok=True)

def key_for(payload: dict) -> str:
    """Stable key for payload to cache response."""
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.json")

async def read_cache(path: str):
    if not os.path.exists(path):
        return None
    async with aiofiles.open(path, "r") as f:
        return await f.read()

async def write_cache(path: str, text: str):
    tmp = path + ".tmp"
    async with aiofiles.open(tmp, "w") as f:
        await f.write(text)
    os.replace(tmp, path)  # atomic

def build_payload(prompt: str) -> dict:
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "stream": False
    }

async def one_request(client: httpx.AsyncClient, sem: asyncio.Semaphore, payload: dict):
    path = key_for(payload)
    cached = await read_cache(path)
    if cached is not None:
        return "cache", cached

    # bounded concurrency
    async with sem:
        # retry with exponential backoff for 429/5xx
        delay = 0.25
        for attempt in range(6):
            try:
                r = await client.post(VLLM_URL, json=payload, timeout=TIMEOUT_S)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError("server busy", request=r.request, response=r)
                r.raise_for_status()
                await write_cache(path, r.text)
                return "net", r.text
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError):
                if attempt == 5:
                    raise
                # jittered backoff
                await asyncio.sleep(delay + random.random() * 0.25)
                delay = min(delay * 2, 4.0)

async def main():
    prompts = [f"Say hello #{i} in one sentence." for i in range(REQUESTS)]
    payloads = [build_payload(p) for p in prompts]
    sem = asyncio.Semaphore(CONCURRENCY)

    limits = httpx.Limits(max_connections=CONCURRENCY * 2, max_keepalive_connections=CONCURRENCY)
    timeout = httpx.Timeout(TIMEOUT_S)

    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {API_KEY}"},
        http2=True,  # enables multiplexing & fewer TCP handshakes
        limits=limits,
        timeout=timeout,
    ) as client:
        t0 = time.perf_counter()
        tasks = [one_request(client, sem, pl) for pl in payloads]
        results = await asyncio.gather(*tasks)
        dt = time.perf_counter() - t0

    hits = sum(1 for src, _ in results if src == "cache")
    print(f"Done {len(results)} reqs in {dt:.2f}s "
          f"(~{len(results)/dt:.1f} req/s). Cache hits: {hits}")

if __name__ == "__main__":
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass
    asyncio.run(main())
