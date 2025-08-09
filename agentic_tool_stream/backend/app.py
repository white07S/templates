from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.sqlite import SqliteStorage
import httpx, json, os, asyncio
from typing import AsyncGenerator
from datetime import datetime, timezone         # NEW

from tinydb import TinyDB, Query

# ------------------------------------------------------------------
# File system setup
# ------------------------------------------------------------------
os.makedirs("tmp", exist_ok=True)
DB_PATH = "tmp/user_data.json"
db = TinyDB(DB_PATH)
sessions_table = db.table("sessions")      # one document per (user_id, session_id)
SessionQ = Query()

# ------------------------------------------------------------------
# Hacker-News helper
# ------------------------------------------------------------------
def get_top_hackernews_stories(num_stories: int = 10) -> str:
    resp = httpx.get('https://hacker-news.firebaseio.com/v0/topstories.json')
    story_ids = resp.json()
    stories = []
    for sid in story_ids[:num_stories]:
        s = httpx.get(f'https://hacker-news.firebaseio.com/v0/item/{sid}.json').json()
        s.pop("text", None)
        stories.append(s)
    return json.dumps(stories)

# ------------------------------------------------------------------
# FastAPI + data models
# ------------------------------------------------------------------
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React dev server
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Explicitly include OPTIONS
    allow_headers=["*"],
    expose_headers=["*"],  # Add this to expose all headers
)

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    query: str

class UserRequest(BaseModel):
    user_id: str

class SessionRequest(BaseModel):
    session_id: str

# ------------------------------------------------------------------
# Agent factory (unchanged)
# ------------------------------------------------------------------
def get_agent(session_id: str) -> Agent:
    return Agent(
        model=OpenAIChat(id="gpt-4o"),
        add_history_to_messages=True,
        num_history_responses=3,
        session_id=session_id,
        storage=SqliteStorage(table_name="agent_sessions", db_file="tmp/data.db"),
        description="You are a helpful assistant that always responds in a polite, upbeat and positive manner.",
        tools=[get_top_hackernews_stories],
        show_tool_calls=True
    )

# ------------------------------------------------------------------
# Streaming helper — now logs each turn to TinyDB with a timestamp
# ------------------------------------------------------------------
# ... keep your imports and setup ...

def _sse(data: dict) -> str:
    # One SSE message
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

# ------------------------------------------------------------------
# Streaming helper — SSE + async arun
# ------------------------------------------------------------------
async def generate_stream(agent: Agent, query: str,
                          user_id: str, session_id: str):
    response_acc = ""
    # NOTE: async streaming (doesn't block the event loop)
    response_stream = await agent.arun(query, stream=True)

    try:
        async for event in response_stream:
            et = getattr(event, "event", None)

            if et == "ToolCallStarted":
                yield _sse({"type": "tool_start", "name": event.tool.tool_name})
            elif et == "ToolCallCompleted":
                yield _sse({"type": "tool_end", "name": event.tool.tool_name})
            elif et == "RunResponseContent" and getattr(event, "content", None):
                response_acc += event.content
                # stream as small deltas
                yield _sse({"type": "content", "delta": event.content})
    finally:
        # persist this turn exactly as you had it
        timestamp = datetime.now(timezone.utc).isoformat()
        turn = {"timestamp": timestamp, "query": query, "response": response_acc}

        existing = sessions_table.get(
            (SessionQ.session_id == session_id) & (SessionQ.user_id == user_id)
        )
        if existing:
            sessions_table.update(
                {"conversation": existing["conversation"] + [turn]},
                doc_ids=[existing.doc_id]
            )
        else:
            sessions_table.insert(
                {"user_id": user_id, "session_id": session_id, "conversation": [turn]}
            )
        # tell client we're done
        yield _sse({"type": "done"})

# ------------------------------------------------------------------
# API endpoint — return proper SSE
# ------------------------------------------------------------------
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        agent = get_agent(req.session_id)
        gen = generate_stream(agent, req.query, req.user_id, req.session_id)
        return StreamingResponse(
            gen,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # If you ever sit behind nginx, this helps disable buffering:
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/get-all-session-id")
async def get_all_session_id(req: UserRequest):
    docs = sessions_table.search(SessionQ.user_id == req.user_id)
    return {"session_ids": [d["session_id"] for d in docs]}

@app.post("/get-chat-details")
async def get_chat_details(req: SessionRequest):
    doc = sessions_table.get(SessionQ.session_id == req.session_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"conversation": doc["conversation"]}
