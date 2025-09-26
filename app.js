import requests

URL = "http://localhost:8002/v1/rerank"  # adjust port if needed
HEADERS = {
    "Authorization": "Bearer dummy",
    "Content-Type": "application/json",
}

payload = {
    "model": "BAAI/bge-reranker-v2-m3",     # optional; server default is fine
    "query": "what is vLLM used for?",
    "documents": [
        "vLLM is a fast LLM inference engine optimized for throughput.",
        "The capital of France is Paris.",
        "Cats are small domesticated carnivores.",
    ],
    "top_n": 3,                 # return top-k results
    "return_documents": True,   # include matched document text in results
}

resp = requests.post(URL, headers=HEADERS, json=payload, timeout=30)
resp.raise_for_status()
data = resp.json()

# Expected shape: {"results": [{"index": int, "relevance_score": float, "document": "..."}], ...}
for i, r in enumerate(sorted(data["results"], key=lambda x: -x["relevance_score"])):
    print(f"{i+1}. score={r['relevance_score']:.4f}  idx={r['index']}  doc={r.get('document')}")
