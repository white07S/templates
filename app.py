from __future__ import annotations
import asyncio, json
from typing import List, Optional
import httpx
from pydantic import BaseModel, Field

# ---- 1) Define your Pydantic schema (Pydantic v2) ----
class Finding(BaseModel):
    control_id: str = Field(..., description="Identifier of the control")
    risk: str = Field(..., description="Short name of the risk")
    severity: str = Field(..., description="low|medium|high|critical")
    rationale: str = Field(..., description="Why the severity is assigned")

class RiskReport(BaseModel):
    summary: str
    findings: List[Finding]
    recommended_actions: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)

# ---- 2) Build request payload with JSON schema from Pydantic ----
def make_payload(prompt: str, model: str) -> dict:
    schema = RiskReport.model_json_schema()  # Pydantic v2
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": (
                "You MUST return ONLY valid JSON that conforms to the provided JSON Schema. "
                "Do not include any prose before or after the JSON."
            )},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 1024,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "risk_report",
                "schema": schema,
                "strict": True,  # reject nonconforming output
            }
        },
        # Optional: improve determinism
        "seed": 1,
        "top_p": 1
    }

# ---- 3) Async httpx client call ----
async def generate_structured(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: float = 60.0
) -> RiskReport:
    payload = make_payload(prompt, model)
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        r = await client.post("/v1/chat/completions", json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        # Validate & parse with Pydantic for safety
        return RiskReport.model_validate_json(content)

# ---- 4) Example usage ----
async def main():
    base_url = "http://localhost:8000"  # vLLM OpenAI server
    api_key = "EMPTY"                   # vLLM ignores by default unless configured
    model = "qwen2.5-7b-instruct"       # example model served by vLLM
    prompt = (
        "Given the application 'Retail Payments Gateway', list top risks found in the last audit "
        "and recommended actions."
    )
    report = await generate_structured(base_url, api_key, model, prompt)
    print(report.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(main())
