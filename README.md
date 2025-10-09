You’re describing a classic latency-vs-accuracy trade-off in tool routing. Don’t let a “planner LLM” sit on the hot path for trivial turns. Use a staged router with early-exit and confidence thresholds.

Here’s a design that works well in practice and avoids “auto tool choice” while still being snappy:

0) The principle
	•	Fast path first: cheap, deterministic checks handle greetings/acks/short chit-chat.
	•	Shallow scoring second: lightweight lexical + keyword scoring proposes 0–N tool candidates.
	•	LLM planner last (only when ambiguous or multi-tool): a small model decides final need_tools and the minimal set/order.

1) Minimal features to extract (cheap)

From the normalized user text:
	•	tok_len, has_question, has_code_block, has_url, has_numbers
	•	imperatives_hit (e.g., fetch, search, compare, summarize, extract, translate, convert, call, run)
	•	tool_keywords_hit[tool] (per-tool synonym lexicon)
	•	Conversation state flags (e.g., “awaiting data upload”, “previous tool pending”)

2) Early-exit guard (no LLM)

If all:
	•	tok_len ≤ 3 and
	•	matches a small-talk set (hi/hello/thanks/ok/👍/bye/etc.) and
	•	not in any tool keyword whitelist
→ NO_TOOLS (reply directly).

Also early-exit WITH tool for obvious patterns:
	•	Has URL → web_retriever or pdf_loader
	•	Has code block → code_runner or linter
	•	Starts with verbs like “translate/summarize/extract” → respective single tool

3) Shallow tool scoring (no LLM)

Score each tool with a cheap hybrid:

score(tool) = 0.5 * Jaccard(query_tokens, tool.desc_tokens)
            + 0.4 * keyword_hits(tool)
            + 0.1 * char_ngrams_overlap

Pick Top-K where score ≥ θ_low.
	•	If none ≥ θ_low → NO_TOOLS (pure chat/QA).
	•	If one ≥ θ_high and tok_len < short_cap → ONE_TOOL (skip planner).
	•	Else → ambiguous → go to planner.

Use gap test: if (top1 − top2) ≥ Δ, accept top1 without planner.

4) Cheap LLM planner (only when needed)

Prompt a small model (local 3B–7B is fine) with only the query + the Top-K tool cards (name, purpose, inputs, examples). Constrain to a strict JSON schema:

{
  "need_tools": true,
  "tools": [
    {"name": "pdf_loader", "why": "query includes a PDF url"},
    {"name": "ner_extractor", "why": "user asked to extract entities"}
  ],
  "order": ["pdf_loader", "ner_extractor"],
  "notes": "Return minimal spans only."
}

Put tight caps on max_tokens and a short timeout; if it times out, fall back to Top-1 from the shallow scorer.

5) Caching & state
	•	Cache the router output for the last N turns per user (norm(query) → decision) with short TTL.
	•	If the previous turn involved a tool and the new turn is anaphoric (“and for 2023?”), reuse previous tool unless shallow scorer contradicts.

6) Example implementation (to-the-point, no external deps)

import re
from typing import List, Dict, Any, Tuple

SMALLTALK = {
    "hi","hello","hey","thanks","thank you","ok","okay","cool","great",
    "nice","yo","sup","bye","goodbye","see ya","ciao","bravo","👍","👌","👋"
}
IMPERATIVES = {"fetch","search","compare","summarize","extract","translate","convert","run","call","plot","download"}

Tool = Dict[str, Any]
tools_catalog: List[Tool] = [
    {
        "name": "web_retriever",
        "desc": "Search and fetch web pages or APIs by URL or query, then return text.",
        "keywords": {"http","https","url","web","site","search","google","fetch"}
    },
    {
        "name": "pdf_loader",
        "desc": "Load a PDF from a URL or bytes and return clean text and pages.",
        "keywords": {"pdf","document","pages","file","report"}
    },
    {
        "name": "code_runner",
        "desc": "Execute user-provided code safely and return stdout/stderr.",
        "keywords": {"code","python","run","execute","stacktrace","error"}
    },
    {
        "name": "ner_extractor",
        "desc": "Extract entities (people, orgs, places) from given text.",
        "keywords": {"extract","entities","people","orgs","locations","named","NER"}
    },
    # add more tools...
]

def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

def char_ngrams(s: str, n=3) -> set:
    s = re.sub(r"\s+", " ", s)
    return {s[i:i+n] for i in range(len(s)-n+1)} if len(s) >= n else set()

def shallow_scores(query: str, catalog: List[Tool]) -> List[Tuple[str, float]]:
    q_toks = tokens(query)
    q_ngr = char_ngrams(query)
    out = []
    for t in catalog:
        desc_toks = tokens(t["desc"])
        kw_hits = len(set(q_toks) & t["keywords"])
        score = (
            0.5 * jaccard(q_toks, desc_toks) +
            0.4 * (kw_hits > 0) +
            0.1 * (len(q_ngr & char_ngrams(t["desc"])) > 0)
        )
        out.append((t["name"], float(score)))
    return sorted(out, key=lambda x: x[1], reverse=True)

def early_exit(query: str) -> Dict[str, Any] | None:
    q = norm(query)
    if len(tokens(q)) <= 3 and (q in SMALLTALK or any(q == s or q.startswith(s) for s in SMALLTALK)):
        return {"need_tools": False, "why": "smalltalk", "tools": []}
    if "```" in query or "code" in tokens(q):
        return {"need_tools": True, "why": "code block present", "tools": ["code_runner"], "order": ["code_runner"]}
    if "http://" in q or "https://" in q:
        # crude PDF vs web split
        return {"need_tools": True, "why": "url present", "tools": (["pdf_loader"] if ".pdf" in q else ["web_retriever"]), "order": (["pdf_loader"] if ".pdf" in q else ["web_retriever"])}
    # imperative single-intent fast path
    if any(w in q for w in IMPERATIVES) and len(tokens(q)) < 15:
        # let shallow scorer pick the one tool
        return None
    return None

def route(query: str, θ_low=0.15, θ_high=0.45, Δ=0.20, short_cap=12) -> Dict[str, Any]:
    ee = early_exit(query)
    if ee is not None:
        return ee

    scores = shallow_scores(query, tools_catalog)
    top1, s1 = scores[0]
    top2, s2 = scores[1] if len(scores) > 1 else (None, 0.0)

    if s1 < θ_low:
        return {"need_tools": False, "why": "scores below θ_low", "tools": []}

    q_len = len(tokens(query))
    if (s1 >= θ_high and (s1 - s2) >= Δ) or (q_len <= short_cap and s1 >= θ_high):
        return {"need_tools": True, "why": "confident single-tool match", "tools": [top1], "order": [top1], "scores": scores[:3]}

    # Ambiguous → hand off to planner (small LLM) with only Top-K tool cards
    topK = [name for name, _ in scores[:3]]
    return {
        "need_tools": "planner",
        "why": "ambiguous tool choice; escalate to small planner",
        "candidates": topK,
        "scores": scores[:3],
        "planner_schema": {
            "type": "object",
            "properties": {
                "need_tools": {"type":"boolean"},
                "tools": {"type":"array","items":{"type":"object","properties":{"name":{"type":"string"},"why":{"type":"string"}}}},
                "order": {"type":"array","items":{"type":"string"}},
                "notes": {"type":"string"}
            },
            "required": ["need_tools","tools","order"]
        }
    }

# --- quick demos ---
# print(route("hello"))
# print(route("fetch this https://example.com/abc.pdf"))
# print(route("extract people and orgs from this text"))

What you get with this:
	•	“hello” → immediate NO_TOOLS.
	•	URL/PDF/code → immediate ONE_TOOL.
	•	Short imperative with clear keywords → ONE_TOOL via shallow scorer.
	•	Confusing/multi-intent → planner only then.

7) Practical knobs
	•	Start with θ_low=0.15, θ_high=0.45, Δ=0.20, short_cap=12. Tune on your logs.
	•	Maintain a per-tool synonym lexicon—that’s the highest ROI.
	•	Put a hard time budget around the planner; on timeout, pick Top-1.

8) When to invoke full ReAct

Gate ReAct behind:
	•	tok_len ≥ 15 and (has_question or imperatives_hit ≥ 1)
	•	More than one tool required or tool requires multi-step transformation (e.g., pdf_loader → ner_extractor → web_retriever).
Otherwise, do a single tool call with a constrained prompt and skip scratch-pad reasoning.

9) Upgrades (when you want more accuracy without latency pain)
	•	Replace shallow scorer with BM25 + embeddings combo and RRF ranker (still cheap).
	•	Add tool co-usage graph to avoid silly multi-tool combos (e.g., pdf_loader pairs with ner_extractor, not code_runner).
	•	Add per-user short-term memory to resolve anaphora and keep the same tool for follow-ups.

This gives you a crisp, low-latency router that avoids burning LLM cycles for “hello,” but still escalates gracefully when a real multi-tool plan is needed.