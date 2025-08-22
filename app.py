Here’s a clean setup that works with vLLM’s reranker API (OpenAI-compatible server) and Qwen’s 8B reranker.

1) Serve the model (GPU box)

# vLLM >= 0.8.3 recommended
pip install -U vllm

# Run the OpenAI-compatible server with the reranker task
vllm serve Qwen/Qwen3-Reranker-8B \
  --task rerank \
  --dtype auto \
  --trust-remote-code \
  --port 8000 \
  --api-key token-abc123

Notes: vLLM provides /v1/rerank (OpenAI-style) and /rerank (Cohere-compatible) endpoints when --task rerank is used.  ￼

⸻

2) Quick test via cURL (OpenAI-style /v1/rerank)

curl -s http://localhost:8000/v1/rerank \
  -H "Authorization: Bearer token-abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Reranker-8B",
    "query": "best places to hike near Krakow",
    "documents": [
      {"text": "Zakopane offers many hiking trails in the Tatras."},
      {"text": "Krakow is known for its medieval old town."},
      {"text": "Ojców National Park has short scenic hikes near Krakow."}
    ],
    "top_n": 3,
    "return_documents": true
  }'

Response includes sorted results[i].relevance_score and (if requested) the document text. Endpoint availability and payload shape are documented by vLLM.  ￼

⸻

3) Python test (requests) hitting /v1/rerank

import requests

BASE = "http://localhost:8000"
headers = {"Authorization": "Bearer token-abc123", "Content-Type": "application/json"}

payload = {
    "model": "Qwen/Qwen3-Reranker-8B",
    "query": "best places to hike near Krakow",
    "documents": [
        {"text": "Zakopane offers many hiking trails in the Tatras."},
        {"text": "Krakow is known for its medieval old town."},
        {"text": "Ojców National Park has short scenic hikes near Krakow."}
    ],
    "top_n": 2,
    "return_documents": True
}

r = requests.post(f"{BASE}/v1/rerank", headers=headers, json=payload, timeout=60)
r.raise_for_status()
for idx, item in enumerate(r.json()["results"], 1):
    print(idx, round(item["relevance_score"], 4), item["document"]["text"])

vLLM’s rerank API sorts and returns scores; top_n is required by several clients.  ￼ ￼

⸻

4) Cohere-compatible client (same server, /rerank)

# pip install cohere
import cohere
client = cohere.Client(api_key="token-abc123", base_url="http://localhost:8000")

out = client.rerank(
    model="Qwen/Qwen3-Reranker-8B",
    query="best places to hike near Krakow",
    documents=[
        "Zakopane offers many hiking trails in the Tatras.",
        "Krakow is known for its medieval old town.",
        "Ojców National Park has short scenic hikes near Krakow."
    ],
    top_n=2
)
for r in out.results:
    print(r.index, r.relevance_score, r.document)

vLLM exposes a Cohere-compatible /rerank endpoint; the official example uses the Cohere SDK exactly like this.  ￼ ￼

⸻

References
	•	Qwen/Qwen3-Reranker-8B model card & notes.  ￼
	•	vLLM OpenAI-compatible server & rerank endpoints.  ￼
	•	vLLM example: Cohere rerank client.  ￼
	•	vLLM example docs for Qwen3 reranker (offline inference details).  ￼

If you want a minimal latency profile, pin --max-num-seqs and --gpu-memory-utilization, and consider TP if you’re on multi-GPU.
