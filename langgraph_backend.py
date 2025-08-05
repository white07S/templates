import os
import math
import json
import asyncio
import logging
import traceback
from typing import AsyncGenerator, Optional, Dict, Any, List
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
    STREAM_DELAY = float(os.getenv("STREAM_DELAY", "0.01"))  # Delay between chunks for smoother streaming

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

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str

# Session Management (in production, use Redis or similar)
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, List[BaseMessage]] = {}
        self.max_sessions = 1000
        self.max_messages_per_session = 100
    
    def get_session(self, session_id: str) -> List[BaseMessage]:
        return self.sessions.get(session_id, [])
    
    def update_session(self, session_id: str, messages: List[BaseMessage]):
        if len(self.sessions) >= self.max_sessions and session_id not in self.sessions:
            # Remove oldest session (simple FIFO)
            oldest = next(iter(self.sessions))
            del self.sessions[oldest]
            logger.info(f"Removed oldest session: {oldest}")
        
        # Trim messages if too many
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
    """Perform basic mathematical calculations. 
    Input should be a valid Python mathematical expression."""
    try:
        # Sanitize input
        allowed_chars = set('0123456789+-*/()., ')
        allowed_funcs = ['sin', 'cos', 'tan', 'sqrt', 'log', 'exp', 'pow', 'abs']
        
        # Check for any suspicious characters
        if not all(c in allowed_chars or any(func in expression for func in allowed_funcs) for c in expression):
            return "Error: Invalid characters in expression"
        
        # Use eval safely for basic math operations
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
            "langchain": "LangChain is a framework for developing applications powered by language models. It provides tools for prompt management, chains, data augmented generation, agents, memory, and evaluation.",
            "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with graph-based workflow capabilities.",
            "vllm": "vLLM is a fast and easy-to-use library for LLM inference and serving. It achieves high throughput with PagedAttention and continuous batching.",
            "fastapi": "FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints.",
        }
        
        query_lower = query.lower()
        results = []
        for key, value in search_results.items():
            if key in query_lower:
                results.append(value)
        
        if results:
            response = " ".join(results)
        else:
            response = f"Search results for '{query}': General web search would be performed here in production."
        
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

# LangGraph Agent Class
class LangGraphAgent:
    def __init__(self):
        self.llm = None
        self.app = None
        self.initialize()
    
    def initialize(self):
        """Initialize the LLM and build the graph."""
        try:
            # Initialize LLM
            self.llm = ChatOpenAI(
                base_url=config.VLLM_BASE_URL,
                api_key=config.VLLM_API_KEY,
                model=config.MODEL_NAME,
                temperature=config.TEMPERATURE,
                streaming=True,  # Enable streaming
                max_retries=config.MAX_RETRIES,
                request_timeout=config.TIMEOUT,
            )
            
            # Bind tools to LLM
            llm_with_tools = self.llm.bind_tools(tools)
            
            # Build the graph
            workflow = StateGraph(AgentState)
            
            # Add nodes
            workflow.add_node("agent", lambda state: self.call_model(state, llm_with_tools))
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
    
    def call_model(self, state: AgentState, llm_with_tools):
        """Call the model with the current state."""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(self, state: AgentState):
        """Determine whether to continue with tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "__end__"
    
    async def process_streaming(
        self, 
        message: str, 
        session_id: str, 
        temperature: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        """Process message with streaming response."""
        try:
            # Get session history
            history = session_manager.get_session(session_id)
            
            # Create initial state
            initial_state = {
                "messages": history + [HumanMessage(content=message)]
            }
            
            # Track tools used
            tools_used = []
            
            # Configure temperature if provided
            if temperature is not None:
                self.llm.temperature = temperature
            
            # Stream the response
            config = RunnableConfig(
                callbacks=[],
                tags=["streaming"],
                metadata={"session_id": session_id}
            )
            
            # Process with the graph
            async for event in self.app.astream_events(
                initial_state,
                config=config,
                version="v2"
            ):
                kind = event["event"]
                
                if kind == "on_chat_model_stream":
                    # Stream content chunks
                    content = event["data"]["chunk"].content
                    if content:
                        yield f"data: {json.dumps({'type': 'content', 'data': content})}\n\n"
                        await asyncio.sleep(config.STREAM_DELAY)  # Small delay for smoother streaming
                
                elif kind == "on_tool_start":
                    # Tool execution started
                    tool_name = event["name"]
                    tools_used.append(tool_name)
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_name})}\n\n"
                
                elif kind == "on_tool_end":
                    # Tool execution completed
                    tool_name = event["name"]
                    tool_output = event["data"].get("output", "")
                    yield f"data: {json.dumps({'type': 'tool_end', 'tool': tool_name, 'output': tool_output})}\n\n"
            
            # Get final state and update session
            final_state = await self.app.ainvoke(initial_state, config)
            session_manager.update_session(session_id, final_state["messages"])
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'done', 'tools_used': tools_used})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}\n{traceback.format_exc()}")
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
            
            # Configure temperature if provided
            if temperature is not None:
                self.llm.temperature = temperature
            
            # Process with the graph
            result = await self.app.ainvoke(initial_state)
            
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

# FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting FastAPI application...")
    yield
    # Shutdown
    logger.info("Shutting down FastAPI application...")

app = FastAPI(
    title="LangGraph vLLM Streaming API",
    description="FastAPI application for LangGraph with vLLM backend and streaming support",
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

# Health check endpoint
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

# Chat endpoint with streaming
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat responses using Server-Sent Events."""
    try:
        # Generate session ID if not provided
        if not request.session_id:
            import uuid
            request.session_id = str(uuid.uuid4())
        
        logger.info(f"Streaming chat request - Session: {request.session_id}, Message: {request.message[:50]}...")
        
        # Create streaming response
        return StreamingResponse(
            agent.process_streaming(
                request.message,
                request.session_id,
                request.temperature
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Session-Id": request.session_id
            }
        )
        
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Stream endpoint error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

# Chat endpoint without streaming
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat message without streaming."""
    try:
        # Generate session ID if not provided
        if not request.session_id:
            import uuid
            request.session_id = str(uuid.uuid4())
        
        logger.info(f"Chat request - Session: {request.session_id}, Message: {request.message[:50]}...")
        
        # Process message
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
        
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

# Clear session endpoint
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

# Get session history endpoint
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
                    "content": msg.content
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

# Error handler for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")
    return {
        "error": "Internal server error",
        "detail": str(exc) if app.debug else "An unexpected error occurred",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",  # Update this to match your filename
        host="0.0.0.0",
        port=8080,
        reload=True,  # Set to False in production
        log_level="info",
        access_log=True,
        workers=1  # Increase for production
    )



"""
Example client for testing the FastAPI streaming chat API.
Demonstrates both streaming and non-streaming endpoints.
"""

import json
import httpx
import asyncio
from typing import Optional
import sys

# API Configuration
API_BASE_URL = "http://localhost:8080"
SESSION_ID = None  # Will be set by the server if not provided

async def test_streaming_chat(message: str, session_id: Optional[str] = None):
    """Test the streaming chat endpoint."""
    print(f"\n🚀 Streaming Request: {message}")
    print("-" * 50)
    
    async with httpx.AsyncClient() as client:
        payload = {
            "message": message,
            "session_id": session_id,
            "stream": True
        }
        
        try:
            async with client.stream(
                "POST",
                f"{API_BASE_URL}/chat/stream",
                json=payload,
                timeout=30.0
            ) as response:
                response.raise_for_status()
                
                # Get session ID from headers
                session_id = response.headers.get("X-Session-Id")
                print(f"📌 Session ID: {session_id}\n")
                
                # Process streaming response
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            
                            if data["type"] == "content":
                                # Print content as it streams
                                print(data["data"], end="", flush=True)
                            
                            elif data["type"] == "tool_start":
                                print(f"\n🔧 Using tool: {data['tool']}", flush=True)
                            
                            elif data["type"] == "tool_end":
                                print(f"✅ Tool result: {data['output'][:100]}...\n", flush=True)
                            
                            elif data["type"] == "done":
                                print(f"\n\n✨ Complete! Tools used: {data['tools_used']}")
                            
                            elif data["type"] == "error":
                                print(f"\n❌ Error: {data['error']}")
                        
                        except json.JSONDecodeError:
                            pass
                
                return session_id
                
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP Error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    return session_id

async def test_regular_chat(message: str, session_id: Optional[str] = None):
    """Test the non-streaming chat endpoint."""
    print(f"\n📨 Regular Request: {message}")
    print("-" * 50)
    
    async with httpx.AsyncClient() as client:
        payload = {
            "message": message,
            "session_id": session_id,
            "stream": False
        }
        
        try:
            response = await client.post(
                f"{API_BASE_URL}/chat",
                json=payload,
                timeout=30.0
            )
            response.raise_for_status()
            
            data = response.json()
            print(f"📌 Session ID: {data['session_id']}")
            print(f"🔧 Tools used: {data['tools_used']}")
            print(f"📝 Response: {data['response']}")
            print(f"⏰ Timestamp: {data['timestamp']}")
            
            return data['session_id']
            
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP Error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    return session_id

async def get_session_history(session_id: str):
    """Get the history of a session."""
    print(f"\n📜 Session History for: {session_id}")
    print("-" * 50)
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{API_BASE_URL}/session/{session_id}/history",
                timeout=10.0
            )
            response.raise_for_status()
            
            data = response.json()
            print(f"Total messages: {data['count']}")
            
            for i, msg in enumerate(data['messages'], 1):
                print(f"{i}. [{msg['type']}]: {msg['content'][:100]}...")
            
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP Error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

async def clear_session(session_id: str):
    """Clear a session."""
    print(f"\n🗑️ Clearing session: {session_id}")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.delete(
                f"{API_BASE_URL}/session/{session_id}",
                timeout=10.0
            )
            response.raise_for_status()
            
            data = response.json()
            print(f"✅ {data['message']}")
            
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP Error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

async def health_check():
    """Check API health."""
    print("\n🏥 Health Check")
    print("-" * 50)
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{API_BASE_URL}/health",
                timeout=5.0
            )
            response.raise_for_status()
            
            data = response.json()
            print(f"✅ Status: {data['status']}")
            print(f"📊 Model: {data['config']['model']}")
            print(f"🔗 vLLM URL: {data['config']['vllm_url']}")
            print(f"⏰ Timestamp: {data['timestamp']}")
            
            return True
            
        except Exception as e:
            print(f"❌ API is not available: {str(e)}")
            return False

async def run_demo():
    """Run a complete demo of the API."""
    print("=" * 60)
    print("🎯 FastAPI LangGraph Streaming Demo")
    print("=" * 60)
    
    # Check health first
    if not await health_check():
        print("\n⚠️ Please ensure the API is running on http://localhost:8080")
        return
    
    # Test queries
    test_queries = [
        "What's the weather like in New York and London?",
        "Calculate the square root of 144 plus 10",
        "Search for information about LangGraph and vLLM",
        "Hello! Can you help me with both: 1) What's 25 * 4? and 2) Search for FastAPI?",
    ]
    
    # Test streaming
    print("\n" + "=" * 60)
    print("📡 STREAMING TESTS")
    print("=" * 60)
    
    session_id = None
    for query in test_queries[:2]:
        session_id = await test_streaming_chat(query, session_id)
        await asyncio.sleep(1)
    
    # Test non-streaming
    print("\n" + "=" * 60)
    print("📨 NON-STREAMING TESTS")
    print("=" * 60)
    
    for query in test_queries[2:]:
        session_id = await test_regular_chat(query, session_id)
        await asyncio.sleep(1)
    
    # Get session history
    if session_id:
        await get_session_history(session_id)
    
    # Clear session
    if session_id:
        await clear_session(session_id)

async def interactive_mode():
    """Run in interactive mode."""
    print("\n" + "=" * 60)
    print("💬 Interactive Mode (type 'quit' to exit)")
    print("Commands: /stream, /regular, /history, /clear, /help")
    print("=" * 60)
    
    session_id = None
    streaming = True
    
    while True:
        try:
            user_input = input("\n You: ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            elif user_input == '/help':
                print("""
Commands:
  /stream   - Switch to streaming mode (default)
  /regular  - Switch to non-streaming mode
  /history  - Show session history
  /clear    - Clear current session
  /help     - Show this help
  quit      - Exit the application
                """)
            
            elif user_input == '/stream':
                streaming = True
                print("✅ Switched to streaming mode")
            
            elif user_input == '/regular':
                streaming = False
                print("✅ Switched to non-streaming mode")
            
            elif user_input == '/history':
                if session_id:
                    await get_session_history(session_id)
                else:
                    print("❌ No active session")
            
            elif user_input == '/clear':
                if session_id:
                    await clear_session(session_id)
                    session_id = None
                else:
                    print("❌ No active session")
            
            elif user_input:
                if streaming:
                    session_id = await test_streaming_chat(user_input, session_id)
                else:
                    session_id = await test_regular_chat(user_input, session_id)
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

async def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        await run_demo()
    else:
        await interactive_mode()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
