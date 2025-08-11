import asyncio, os
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

PROMPTS = [...]  # your 9,000 prompts
CONCURRENCY = 256  # tune: start 128–512

sem = asyncio.Semaphore(CONCURRENCY)

async def one_call(p):
    async with sem:
        r = await client.chat.completions.create(
            model="your-model",
            messages=[{"role":"user","content":p}],
            temperature=0,
            stream=False,   # disable streaming to reduce overhead when bulk-processing
        )
        return r.choices[0].message.content

async def main():
    tasks = [asyncio.create_task(one_call(p)) for p in PROMPTS]
    results = await asyncio.gather(*tasks)
    # write results somewhere
    return results

asyncio.run(main())
