from fastapi import FastAPI, HTTPException, Query as FastAPIQuery
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
from typing import List, Optional
import uuid

# ------------------------------------------------------------------
# File system setup
# ------------------------------------------------------------------
os.makedirs("tmp", exist_ok=True)
DB_PATH = "tmp/user_data.json"
db = TinyDB(DB_PATH)
sessions_table = db.table("sessions")      # one document per (user_id, session_id)
prompts_table = db.table("prompts")       # prompt library
SessionQ = Query()
PromptQ = Query()

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

class PromptCreate(BaseModel):
    persona: str
    task: str
    if_task_need_data: bool = False
    data: Optional[str] = None
    response: str
    keywords_used_for_search: List[str]

class PromptUpdate(BaseModel):
    persona: Optional[str] = None
    task: Optional[str] = None
    if_task_need_data: Optional[bool] = None
    data: Optional[str] = None
    response: Optional[str] = None
    keywords_used_for_search: Optional[List[str]] = None

class PromptFilter(BaseModel):
    user_created: Optional[bool] = None
    keywords: Optional[List[str]] = None

class PromptCopy(BaseModel):
    prompt_id: str

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

# ------------------------------------------------------------------
# Prompt Library API Endpoints
# ------------------------------------------------------------------

@app.post("/prompts/create")
async def create_prompt(req: PromptCreate, user_id: str = FastAPIQuery()):
    try:
        # Validate prompt
        if not req.persona or not req.task or not req.response:
            raise HTTPException(status_code=400, detail="Persona, task, and response are required")
        
        if req.if_task_need_data and not req.data:
            raise HTTPException(status_code=400, detail="Data is required when if_task_need_data is true")
        
        prompt_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        prompt_doc = {
            "id": prompt_id,
            "user_id": user_id,
            "persona": req.persona,
            "task": req.task,
            "if_task_need_data": req.if_task_need_data,
            "data": req.data,
            "response": req.response,
            "keywords_used_for_search": req.keywords_used_for_search,
            "created_at": timestamp,
            "updated_at": timestamp
        }
        
        prompts_table.insert(prompt_doc)
        return {"message": "Prompt created successfully", "prompt_id": prompt_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/prompts/list")
async def list_prompts(user_created: Optional[bool] = None, keywords: Optional[str] = None, user_id: str = FastAPIQuery()):
    try:
        # Start with all prompts
        all_prompts = prompts_table.all()
        
        # Filter by user_created if specified
        if user_created is not None:
            if user_created:
                all_prompts = [p for p in all_prompts if p.get("user_id") == user_id]
            else:
                all_prompts = [p for p in all_prompts if p.get("user_id") != user_id]
        
        # Filter by keywords if specified
        if keywords:
            keyword_list = [k.strip().lower() for k in keywords.split(",")]
            filtered_prompts = []
            for prompt in all_prompts:
                prompt_keywords = [k.lower() for k in prompt.get("keywords_used_for_search", [])]
                if any(kw in prompt_keywords for kw in keyword_list):
                    filtered_prompts.append(prompt)
            all_prompts = filtered_prompts
        
        # Add is_owner flag to each prompt
        for prompt in all_prompts:
            prompt["is_owner"] = prompt.get("user_id") == user_id
        
        return {"prompts": all_prompts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/prompts/{prompt_id}")
async def get_prompt(prompt_id: str, user_id: str = FastAPIQuery()):
    try:
        prompt = prompts_table.get(PromptQ.id == prompt_id)
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        
        prompt["is_owner"] = prompt.get("user_id") == user_id
        return {"prompt": prompt}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/prompts/{prompt_id}")
async def update_prompt(prompt_id: str, req: PromptUpdate, user_id: str = FastAPIQuery()):
    try:
        prompt = prompts_table.get(PromptQ.id == prompt_id)
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        
        # Check ownership
        if prompt.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="You can only edit your own prompts")
        
        # Update fields
        update_data = {}
        if req.persona is not None:
            update_data["persona"] = req.persona
        if req.task is not None:
            update_data["task"] = req.task
        if req.if_task_need_data is not None:
            update_data["if_task_need_data"] = req.if_task_need_data
        if req.data is not None:
            update_data["data"] = req.data
        if req.response is not None:
            update_data["response"] = req.response
        if req.keywords_used_for_search is not None:
            update_data["keywords_used_for_search"] = req.keywords_used_for_search
        
        if update_data:
            update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
            prompts_table.update(update_data, PromptQ.id == prompt_id)
        
        return {"message": "Prompt updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/prompts/{prompt_id}")
async def delete_prompt(prompt_id: str, user_id: str = FastAPIQuery()):
    try:
        prompt = prompts_table.get(PromptQ.id == prompt_id)
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        
        # Check ownership
        if prompt.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="You can only delete your own prompts")
        
        prompts_table.remove(PromptQ.id == prompt_id)
        return {"message": "Prompt deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/prompts/copy")
async def copy_prompt(req: PromptCopy, user_id: str = FastAPIQuery()):
    try:
        original = prompts_table.get(PromptQ.id == req.prompt_id)
        if not original:
            raise HTTPException(status_code=404, detail="Prompt not found")
        
        # Create a copy with new ID and ownership
        new_prompt_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        copied_prompt = {
            "id": new_prompt_id,
            "user_id": user_id,
            "persona": original["persona"],
            "task": original["task"],
            "if_task_need_data": original["if_task_need_data"],
            "data": original.get("data"),
            "response": original["response"],
            "keywords_used_for_search": original["keywords_used_for_search"],
            "created_at": timestamp,
            "updated_at": timestamp,
            "copied_from": req.prompt_id
        }
        
        prompts_table.insert(copied_prompt)
        return {"message": "Prompt copied successfully", "prompt_id": new_prompt_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/prompts/suggestions/{limit}")
async def get_prompt_suggestions(limit: int = 5):
    try:
        all_prompts = prompts_table.all()
        # Sort by created_at (newest first) and limit
        all_prompts.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        suggestions = all_prompts[:limit]
        return {"suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))