import aiohttp
import asyncio
import json

async def stream_chat():
    url = "http://localhost:8000/v1/chat/completions"  # Adjust if running on different host/port

    payload = {
        "model": "Qwen/Qwen1.5-0.5B",  # Replace with your model name
        "messages": [
            {"role": "user", "content": "Tell me a joke."}
        ],
        "stream": True,
        "temperature": 0.7
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            async for line in resp.content:
                if line:
                    decoded = line.decode("utf-8").strip()
                    if decoded.startswith("data: "):
                        data = decoded.removeprefix("data: ").strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"]
                            if "content" in delta:
                                print(delta["content"], end="", flush=True)
                        except Exception as e:
                            print(f"\n[Error parsing chunk] {e}\nRaw: {data}")

if __name__ == "__main__":
    asyncio.run(stream_chat())
