import asyncio
import json
import time
from typing import AsyncGenerator, Optional, Dict, Any
from datetime import datetime, timezone
from uuid import UUID
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .models.schemas import ChatRequest, ChatResponse, ChatMessage, MessageRole
from .memory.memory_manager import HybridMemoryManager, MemoryConfig
from .storage.session_manager import SessionManager
from .tools.base_tools import (
    CalculatorTool, FileSearchTool, TimeWeatherTool,
    MemoryTool, NoteTakingTool, ConversationHistoryTool, ToolRegistry
)
from .core.reasoning_agent import ReActAgent
from .utils.llm_client import create_default_llm_manager


# Global instances
llm_manager = None
memory_manager = None
session_manager = None
tool_registry = None
reasoning_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    global llm_manager, memory_manager, session_manager, tool_registry, reasoning_agent

    print("🚀 Starting Chat Backend Server...")

    # Initialize LLM manager
    llm_manager = create_default_llm_manager()
    print("✅ LLM Manager initialized")

    # Initialize memory manager
    memory_manager = HybridMemoryManager(MemoryConfig())
    print("✅ Memory Manager initialized")

    # Initialize session manager
    session_manager = SessionManager()
    print("✅ Session Manager initialized")

    # Initialize tool registry
    tool_registry = ToolRegistry()

    # Register tools
    tool_registry.register_tool(CalculatorTool())
    tool_registry.register_tool(FileSearchTool())
    tool_registry.register_tool(TimeWeatherTool())
    tool_registry.register_tool(MemoryTool(memory_manager))
    tool_registry.register_tool(NoteTakingTool())
    tool_registry.register_tool(ConversationHistoryTool(session_manager))
    print(f"✅ Tools registered: {', '.join(tool_registry.list_tools())}")

    # Initialize reasoning agent
    client = await llm_manager.get_healthy_client()
    if client:
        reasoning_agent = ReActAgent(client, tool_registry)
        print("✅ ReAct Agent initialized")
    else:
        print("⚠️  No healthy LLM client available - reasoning agent disabled")

    # Health check
    health_status = await llm_manager.health_check_all()
    print(f"🏥 LLM Health Status: {health_status}")

    print("🎉 Server startup complete!")

    yield

    # Shutdown
    print("🛑 Shutting down Chat Backend Server...")

    # Cleanup memory
    if memory_manager:
        await memory_manager.garbage_collect()
        print("✅ Memory cleanup completed")

    print("👋 Server shutdown complete!")


# Create FastAPI app
app = FastAPI(
    title="Chat Backend API",
    description="High-performance chat backend with memory, reasoning, and tool calling",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency functions
def get_llm_manager():
    return llm_manager


def get_memory_manager():
    return memory_manager


def get_session_manager():
    return session_manager


def get_tool_registry():
    return tool_registry


def get_reasoning_agent():
    return reasoning_agent


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Chat Backend API", "status": "online", "timestamp": datetime.now(timezone.utc)}


@app.get("/health")
async def health_check(llm_mgr=Depends(get_llm_manager)):
    """Health check endpoint"""
    if not llm_mgr:
        raise HTTPException(status_code=503, detail="LLM Manager not initialized")

    health_status = await llm_mgr.health_check_all()

    return {
        "status": "healthy" if any(h["status"] == "healthy" for h in health_status.values()) else "unhealthy",
        "timestamp": datetime.now(timezone.utc),
        "llm_clients": health_status,
        "components": {
            "memory_manager": "online" if memory_manager else "offline",
            "session_manager": "online" if session_manager else "offline",
            "tool_registry": "online" if tool_registry else "offline",
            "reasoning_agent": "online" if reasoning_agent else "offline"
        }
    }


@app.get("/tools")
async def list_tools(registry=Depends(get_tool_registry)):
    """List available tools"""
    if not registry:
        raise HTTPException(status_code=503, detail="Tool registry not initialized")

    return {
        "tools": registry.get_tools_schema(),
        "count": len(registry.list_tools())
    }


@app.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
    llm_mgr=Depends(get_llm_manager),
    mem_mgr=Depends(get_memory_manager),
    sess_mgr=Depends(get_session_manager),
    registry=Depends(get_tool_registry),
    agent=Depends(get_reasoning_agent)
):
    """Main chat endpoint with streaming support"""

    if not llm_mgr:
        raise HTTPException(status_code=503, detail="LLM Manager not initialized")

    try:
        # Create session if it doesn't exist
        session_info = await sess_mgr.get_session_info(request.user_id, request.session_id)
        if not session_info:
            await sess_mgr.create_session(request.user_id, request.session_id)

        # Get memory hints
        memory_hints = ""
        if mem_mgr:
            memory_hints = await mem_mgr.get_memory_hints(
                request.message,
                request.user_id,
                str(request.session_id) if request.session_id else None
            )

        # Store user message
        user_message = ChatMessage(
            role=MessageRole.USER,
            content=request.message
        )
        await sess_mgr.add_message(request.user_id, request.session_id, user_message)

        # Get recent conversation history
        recent_messages = await sess_mgr.get_session_messages(
            request.user_id,
            request.session_id,
            limit=10
        )

        # Build conversation context
        system_content = """You are a helpful AI assistant with memory and conversation history capabilities.

You have access to the following tools:
- conversation_history: To retrieve previous messages and queries from the current session
- memory_tool: To add or search long-term memories about the user

When users ask about previous queries or conversations, use the conversation_history tool with query_type="last_message" or "recent_messages".
When users want you to remember something important, use the memory_tool with action="add".
"""
        if memory_hints:
            system_content += f"\n{memory_hints}"

        messages = [{"role": "system", "content": system_content}]

        # Add recent conversation history
        for msg in recent_messages[:-1]:  # Exclude the just-added user message
            messages.append({
                "role": msg.role.value,
                "content": msg.content
            })

        # Add current user message
        messages.append({
            "role": "user",
            "content": request.message
        })

        # Handle reasoning mode
        if request.reasoning_mode and agent:
            return StreamingResponse(
                generate_reasoning_stream(
                    request, messages, agent, sess_mgr, mem_mgr
                ),
                media_type="text/plain"
            )
        else:
            # Regular chat mode
            return StreamingResponse(
                generate_chat_stream(
                    request, messages, llm_mgr, sess_mgr, mem_mgr, registry
                ),
                media_type="text/plain"
            )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")


async def generate_chat_stream(
    request: ChatRequest,
    messages: list,
    llm_mgr,
    sess_mgr,
    mem_mgr,
    registry
) -> AsyncGenerator[str, None]:
    """Generate streaming chat response"""
    try:
        # Get tools schema if tools requested
        tools = None
        if request.tools or registry:
            tools = registry.get_tools_schema() if registry else request.tools

        response_content = ""

        # Stream the response
        async for chunk in llm_mgr.create_streaming_completion(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tools=tools
        ):
            response_content += chunk

            # Send chunk in SSE format
            yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

        # Store assistant response
        assistant_message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=response_content
        )
        await sess_mgr.add_message(request.user_id, request.session_id, assistant_message)

        # Extract and store conversation in memory
        if mem_mgr and response_content:
            await mem_mgr.extract_and_store_from_conversation(
                user_message=request.message,
                assistant_message=response_content,
                user_id=request.user_id,
                session_id=request.session_id
            )

        # Send completion signal
        yield f"data: {json.dumps({'type': 'complete', 'message_id': assistant_message.timestamp.isoformat()})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"


async def generate_reasoning_stream(
    request: ChatRequest,
    messages: list,
    agent,
    sess_mgr,
    mem_mgr
) -> AsyncGenerator[str, None]:
    """Generate streaming reasoning response"""
    try:
        # Extract context from recent messages
        context = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in messages[-5:]  # Last 5 messages for context
        ])

        response_content = ""
        reasoning_steps = []

        # Stream reasoning process
        async for step_data in agent.reason_and_act(
            task=request.message,
            context=context,
            user_id=request.user_id,
            session_id=str(request.session_id)
        ):
            # Send step data
            yield f"data: {json.dumps(step_data)}\n\n"

            if step_data["type"] == "final_response":
                response_content = step_data["response"]
                reasoning_steps = step_data["reasoning_steps"]

        # Store assistant response with reasoning metadata
        assistant_message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=response_content,
            metadata={
                "reasoning_mode": True,
                "reasoning_steps": reasoning_steps,
                "step_count": len(reasoning_steps)
            }
        )
        await sess_mgr.add_message(request.user_id, request.session_id, assistant_message)

        # Store reasoning process in memory if significant
        if mem_mgr and len(reasoning_steps) > 2:
            reasoning_summary = f"Solved complex task: {request.message[:100]}... using {len(reasoning_steps)} reasoning steps"
            await mem_mgr.add_memory(
                text=reasoning_summary,
                user_id=request.user_id,
                session_id=request.session_id,
                kind="episodic",
                importance=0.8
            )

        # Send completion
        yield f"data: {json.dumps({'type': 'complete', 'message_id': assistant_message.timestamp.isoformat()})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"


@app.get("/sessions/{user_id}")
async def list_user_sessions(user_id: str, sess_mgr=Depends(get_session_manager)):
    """List all sessions for a user"""
    if not sess_mgr:
        raise HTTPException(status_code=503, detail="Session manager not initialized")

    sessions = await sess_mgr.list_user_sessions(user_id)
    return {"sessions": [session.model_dump() for session in sessions]}


@app.get("/sessions/{user_id}/{session_id}")
async def get_session_messages(
    user_id: str,
    session_id: UUID,
    limit: Optional[int] = 50,
    sess_mgr=Depends(get_session_manager)
):
    """Get messages from a specific session"""
    if not sess_mgr:
        raise HTTPException(status_code=503, detail="Session manager not initialized")

    messages = await sess_mgr.get_session_messages(user_id, session_id, limit)
    return {
        "session_id": session_id,
        "user_id": user_id,
        "messages": [msg.model_dump() for msg in messages],
        "count": len(messages)
    }


@app.get("/memory/{user_id}/stats")
async def get_user_memory_stats(user_id: str, mem_mgr=Depends(get_memory_manager)):
    """Get memory statistics for a user"""
    if not mem_mgr:
        raise HTTPException(status_code=503, detail="Memory manager not initialized")

    stats = await mem_mgr.get_user_memory_stats(user_id)
    return {"user_id": user_id, "memory_stats": stats}


@app.delete("/sessions/{user_id}/{session_id}")
async def delete_session(
    user_id: str,
    session_id: UUID,
    sess_mgr=Depends(get_session_manager)
):
    """Delete a session"""
    if not sess_mgr:
        raise HTTPException(status_code=503, detail="Session manager not initialized")

    success = await sess_mgr.delete_session(user_id, session_id)
    if success:
        return {"message": "Session deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/memory/cleanup")
async def cleanup_memory(mem_mgr=Depends(get_memory_manager)):
    """Trigger memory cleanup/garbage collection"""
    if not mem_mgr:
        raise HTTPException(status_code=503, detail="Memory manager not initialized")

    await mem_mgr.garbage_collect()
    return {"message": "Memory cleanup completed"}


@app.post("/v1/chat/completions")
async def openai_compatible_chat(
    request: Dict[str, Any],
    llm_mgr=Depends(get_llm_manager)
):
    """OpenAI-compatible chat completions endpoint"""

    if not llm_mgr:
        raise HTTPException(status_code=503, detail="LLM Manager not initialized")

    try:
        # Extract parameters from OpenAI format
        messages = request.get("messages", [])
        model = request.get("model", "gpt-4o-mini")
        temperature = request.get("temperature", 0.7)
        max_tokens = request.get("max_tokens", 1000)
        stream = request.get("stream", False)

        if stream:
            # Return streaming response
            return StreamingResponse(
                generate_openai_stream(messages, llm_mgr, temperature, max_tokens),
                media_type="text/event-stream"
            )
        else:
            # Return non-streaming response
            response = await llm_mgr.create_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Format response in OpenAI format
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response.choices[0].message.content
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


async def generate_openai_stream(messages, llm_mgr, temperature, max_tokens):
    """Generate OpenAI-compatible streaming response"""
    try:
        async for chunk in llm_mgr.create_streaming_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        ):
            # Format as OpenAI streaming chunk
            chunk_data = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"

        # Send final chunk
        final_chunk = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )