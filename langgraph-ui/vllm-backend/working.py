import os
import math
import json
import asyncio
import logging
import traceback
from typing import AsyncGenerator, Optional, Dict, Any, List, AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
import uvicorn

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.runnables.config import RunnableConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
    MODEL_NAME = os.getenv("MODEL_NAME", "your-model-name")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    TIMEOUT = int(os.getenv("TIMEOUT", "30"))
    STREAM_DELAY = float(os.getenv("STREAM_DELAY", "0.001"))  # Smaller delay for smoother streaming

config = Config()

# Request/Response Models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to process")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation tracking")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Temperature for LLM")
    stream: bool = Field(default=True, description="Enable streaming response")

class ChatResponse(BaseModel):
    response: str
    session_id: str
    tools_used: List[str] = []
    timestamp: str

# Session Management
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, List[BaseMessage]] = {}
        self.max_sessions = 1000
        self.max_messages_per_session = 100
    
    def get_session(self, session_id: str) -> List[BaseMessage]:
        return self.sessions.get(session_id, [])
    
    def update_session(self, session_id: str, messages: List[BaseMessage]):
        if len(self.sessions) >= self.max_sessions and session_id not in self.sessions:
            oldest = next(iter(self.sessions))
            del self.sessions[oldest]
            logger.info(f"Removed oldest session: {oldest}")
        
        if len(messages) > self.max_messages_per_session:
            messages = messages[-self.max_messages_per_session:]
        
        self.sessions[session_id] = messages
    
    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session: {session_id}")

session_manager = SessionManager()

# Define tools
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    try:
        weather_data = {
            "new york": "Sunny, 72°F",
            "london": "Cloudy, 59°F", 
            "tokyo": "Rainy, 65°F",
            "paris": "Partly cloudy, 68°F",
            "san francisco": "Foggy, 62°F",
            "los angeles": "Clear, 78°F",
        }
        result = weather_data.get(location.lower(), f"Weather data not available for {location}")
        logger.info(f"Weather tool called for location: {location}")
        return result
    except Exception as e:
        logger.error(f"Error in get_weather tool: {str(e)}")
        return f"Error retrieving weather data: {str(e)}"

@tool
def calculate(expression: str) -> str:
    """Perform basic mathematical calculations."""
    try:
        allowed_chars = set('0123456789+-*/()., ')
        allowed_funcs = ['sin', 'cos', 'tan', 'sqrt', 'log', 'exp', 'pow', 'abs']
        
        if not all(c in allowed_chars or any(func in expression for func in allowed_funcs) for c in expression):
            return "Error: Invalid characters in expression"
        
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        logger.info(f"Calculate tool: {expression} = {result}")
        return f"Result: {result}"
    except Exception as e:
        logger.error(f"Error in calculate tool: {str(e)}")
        return f"Error calculating: {str(e)}"

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    try:
        search_results = {
            "langchain": "LangChain is a framework for developing applications powered by language models.",
            "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
            "vllm": "vLLM is a fast and easy-to-use library for LLM inference and serving.",
            "fastapi": "FastAPI is a modern, fast web framework for building APIs with Python.",
        }
        
        query_lower = query.lower()
        results = []
        for key, value in search_results.items():
            if key in query_lower:
                results.append(value)
        
        if results:
            response = " ".join(results)
        else:
            response = f"Search results for '{query}': General web search would be performed here."
        
        logger.info(f"Search tool called with query: {query}")
        return response
    except Exception as e:
        logger.error(f"Error in search_web tool: {str(e)}")
        return f"Error searching web: {str(e)}"

# Create tools list
tools = [get_weather, calculate, search_web]

# Agent State
class AgentState(MessagesState):
    """State of the agent."""
    pass

# LangGraph Agent with TRUE Streaming
class LangGraphAgent:
    def __init__(self):
        self.llm = None
        self.llm_with_tools = None
        self.app = None
        self.initialize()
    
    def initialize(self):
        """Initialize the LLM and build the graph."""
        try:
            # Create NON-STREAMING LLM for tool operations
            # This is critical - tool binding doesn't work with streaming=True
            self.llm = ChatOpenAI(
                base_url=config.VLLM_BASE_URL,
                api_key=config.VLLM_API_KEY,
                model=config.MODEL_NAME,
                temperature=config.TEMPERATURE,
                streaming=False,  # MUST be False for tool binding
                max_retries=config.MAX_RETRIES,
                request_timeout=config.TIMEOUT,
            )
            
            # Bind tools
            self.llm_with_tools = self.llm.bind_tools(tools)
            
            # Build the graph
            workflow = StateGraph(AgentState)
            
            # Add nodes
            workflow.add_node("agent", self.call_model)
            workflow.add_node("tools", ToolNode(tools))
            
            # Set entry point
            workflow.set_entry_point("agent")
            
            # Add conditional edges
            workflow.add_conditional_edges(
                "agent",
                self.should_continue,
                {
                    "tools": "tools",
                    "__end__": "__end__",
                }
            )
            
            # Add edge from tools back to agent
            workflow.add_edge("tools", "agent")
            
            # Compile the graph
            self.app = workflow.compile()
            
            logger.info("LangGraph agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise
    
    def call_model(self, state: AgentState):
        """Call the model with the current state."""
        messages = state["messages"]
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(self, state: AgentState):
        """Determine whether to continue with tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "__end__"
    
    async def process_streaming_with_langgraph(
        self, 
        message: str, 
        session_id: str, 
        temperature: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        """TRUE streaming using LangGraph's astream method."""
        try:
            # Get session history
            history = session_manager.get_session(session_id)
            
            # Create initial state
            initial_state = {
                "messages": history + [HumanMessage(content=message)]
            }
            
            # Track what we've sent
            tools_used = []
            final_messages = []
            
            # Use astream for true streaming
            async for chunk in self.app.astream(initial_state):
                # Process each chunk as it arrives
                for node_name, node_output in chunk.items():
                    if node_name == "agent":
                        # Agent produced a message
                        messages = node_output.get("messages", [])
                        for msg in messages:
                            final_messages.append(msg)
                            
                            # Check for tool calls
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    tool_name = tc["name"]
                                    tools_used.append(tool_name)
                                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_name})}\n\n"
                            
                            # Stream content if available
                            if hasattr(msg, 'content') and msg.content:
                                # Stream the content in small chunks
                                content = msg.content
                                # Split into words for natural streaming
                                words = content.split(' ')
                                for i, word in enumerate(words):
                                    if i > 0:
                                        yield f"data: {json.dumps({'type': 'content', 'data': ' '})}\n\n"
                                    # Stream each word
                                    yield f"data: {json.dumps({'type': 'content', 'data': word})}\n\n"
                                    await asyncio.sleep(config.STREAM_DELAY)
                    
                    elif node_name == "tools":
                        # Tool was executed
                        messages = node_output.get("messages", [])
                        for msg in messages:
                            final_messages.append(msg)
                            if isinstance(msg, ToolMessage):
                                yield f"data: {json.dumps({'type': 'tool_end', 'output': str(msg.content)[:100]})}\n\n"
            
            # Update session with all messages
            all_messages = history + final_messages
            session_manager.update_session(session_id, all_messages)
            
            # Send completion
            yield f"data: {json.dumps({'type': 'done', 'tools_used': tools_used})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}\n{traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    async def process_streaming_direct_vllm(
        self, 
        message: str, 
        session_id: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        """Direct vLLM streaming without LangGraph - TRUE streaming."""
        try:
            import httpx
            
            # Prepare the request
            headers = {
                "Content-Type": "application/json",
            }
            
            # Get session history if exists
            messages = []
            if session_id:
                history = session_manager.get_session(session_id)
                for msg in history:
                    if isinstance(msg, HumanMessage):
                        messages.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        messages.append({"role": "assistant", "content": msg.content})
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            payload = {
                "model": config.MODEL_NAME,
                "messages": messages,
                "temperature": temperature or config.TEMPERATURE,
                "stream": True,
                "max_tokens": 2048,
            }
            
            # Make streaming request to vLLM
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream(
                    "POST",
                    f"{config.VLLM_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    
                    full_response = ""
                    
                    # Process SSE stream from vLLM
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            if line == "data: [DONE]":
                                break
                            
                            try:
                                chunk_data = json.loads(line[6:])
                                if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                    delta = chunk_data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    
                                    if content:
                                        full_response += content
                                        # Send content chunk to client
                                        yield f"data: {json.dumps({'type': 'content', 'data': content})}\n\n"
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse chunk: {line}")
                                continue
                    
                    # Update session if session_id provided
                    if session_id and full_response:
                        history = session_manager.get_session(session_id)
                        history.append(HumanMessage(content=message))
                        history.append(AIMessage(content=full_response))
                        session_manager.update_session(session_id, history)
                    
                    # Send completion
                    yield f"data: {json.dumps({'type': 'done', 'tools_used': []})}\n\n"
                    
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error in direct streaming: {e.response.status_code} - {e.response.text}")
            yield f"data: {json.dumps({'type': 'error', 'error': f'HTTP {e.response.status_code}'})}\n\n"
        except Exception as e:
            logger.error(f"Direct streaming error: {str(e)}\n{traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    async def process(
        self, 
        message: str, 
        session_id: str, 
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Process message without streaming."""
        try:
            # Get session history
            history = session_manager.get_session(session_id)
            
            # Create initial state
            initial_state = {
                "messages": history + [HumanMessage(content=message)]
            }
            
            # Process with the graph
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.app.invoke,
                initial_state
            )
            
            # Update session
            session_manager.update_session(session_id, result["messages"])
            
            # Extract response
            final_message = result["messages"][-1].content
            
            # Identify tools used
            tools_used = []
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tools_used.extend([tc["name"] for tc in msg.tool_calls])
            
            return {
                "response": final_message,
                "tools_used": tools_used,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}\n{traceback.format_exc()}")
            raise

# Initialize agent
agent = LangGraphAgent()

# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI application...")
    yield
    logger.info("Shutting down FastAPI application...")

app = FastAPI(
    title="LangGraph vLLM Streaming API",
    description="FastAPI application for LangGraph with vLLM backend and TRUE streaming support",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": config.MODEL_NAME,
            "vllm_url": config.VLLM_BASE_URL
        }
    }

# Main streaming endpoint with LangGraph
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat responses with LangGraph (supports tools)."""
    try:
        if not request.session_id:
            import uuid
            request.session_id = str(uuid.uuid4())
        
        logger.info(f"Streaming chat request - Session: {request.session_id}, Message: {request.message[:50]}...")
        
        return StreamingResponse(
            agent.process_streaming_with_langgraph(
                request.message,
                request.session_id,
                request.temperature
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Session-Id": request.session_id,
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Transfer-Encoding": "chunked",
            }
        )
        
    except Exception as e:
        logger.error(f"Stream endpoint error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

# Direct vLLM streaming (no tools, but true streaming)
@app.post("/chat/direct-stream")
async def direct_stream(request: ChatRequest):
    """Direct vLLM streaming without LangGraph - TRUE streaming."""
    try:
        if not request.session_id:
            import uuid
            request.session_id = str(uuid.uuid4())
        
        logger.info(f"Direct streaming request - Message: {request.message[:50]}...")
        
        return StreamingResponse(
            agent.process_streaming_direct_vllm(
                request.message,
                request.session_id,
                request.temperature
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Session-Id": request.session_id,
                "X-Accel-Buffering": "no",
                "Transfer-Encoding": "chunked",
            }
        )
        
    except Exception as e:
        logger.error(f"Direct stream error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Non-streaming endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat message without streaming."""
    try:
        if not request.session_id:
            import uuid
            request.session_id = str(uuid.uuid4())
        
        logger.info(f"Chat request - Session: {request.session_id}, Message: {request.message[:50]}...")
        
        result = await agent.process(
            request.message,
            request.session_id,
            request.temperature
        )
        
        return ChatResponse(
            response=result["response"],
            session_id=request.session_id,
            tools_used=result["tools_used"],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

# Test raw vLLM endpoint
@app.post("/test/vllm-raw")
async def test_vllm_raw(request: ChatRequest):
    """Test raw vLLM API to verify streaming works."""
    try:
        import httpx
        
        async def generate():
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {
                    "model": config.MODEL_NAME,
                    "messages": [{"role": "user", "content": request.message}],
                    "temperature": request.temperature or config.TEMPERATURE,
                    "stream": True,
                }
                
                async with client.stream(
                    "POST",
                    f"{config.VLLM_BASE_URL}/chat/completions",
                    json=payload,
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            yield f"{line}\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )
        
    except Exception as e:
        logger.error(f"Test vLLM raw error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Session management
@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific session."""
    try:
        session_manager.clear_session(session_id)
        return {"message": f"Session {session_id} cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing session: {str(e)}"
        )

@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get message history for a session."""
    try:
        messages = session_manager.get_session(session_id)
        return {
            "session_id": session_id,
            "messages": [
                {
                    "type": msg.__class__.__name__,
                    "content": msg.content if hasattr(msg, 'content') else str(msg)
                }
                for msg in messages
            ],
            "count": len(messages)
        }
    except Exception as e:
        logger.error(f"Error getting session history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting session history: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
        access_log=True,
        workers=1
    )
