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
from tinydb import TinyDB, Query

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
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME = os.getenv("MODEL_NAME", "o3-mini")
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    TIMEOUT = int(os.getenv("TIMEOUT", "30"))
    STREAM_DELAY = float(os.getenv("STREAM_DELAY", "0.001"))  # Smaller delay for smoother streaming

config = Config()

# Request/Response Models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to process")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation tracking")
    user_id: Optional[str] = Field(default=None, description="User ID for saving chat history")
    stream: bool = Field(default=True, description="Enable streaming response")

class ChatResponse(BaseModel):
    response: str
    session_id: str
    tools_used: List[str] = []
    timestamp: str

class AuthRequest(BaseModel):
    name: str = Field(..., description="User name for authentication")

class AuthResponse(BaseModel):
    user_id: str
    name: str
    message: str

class UserHistoryResponse(BaseModel):
    user_id: str
    chats: List[Dict[str, Any]]

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

# Database Management
class DatabaseManager:
    def __init__(self):
        self.users_db = TinyDB('users.json')
        self.chats_db = TinyDB('chats.json')
        self.User = Query()
        self.Chat = Query()
    
    def authenticate_user(self, name: str) -> Dict[str, Any]:
        """Simple name-based authentication - checks if user exists, creates if not"""
        user = self.users_db.search(self.User.name == name)
        
        if user:
            user_data = user[0]
            return {
                "user_id": user_data["user_id"],
                "name": user_data["name"],
                "message": "Welcome back!"
            }
        else:
            # Create new user
            import uuid
            user_id = str(uuid.uuid4())
            new_user = {
                "user_id": user_id,
                "name": name,
                "created_at": datetime.now().isoformat()
            }
            self.users_db.insert(new_user)
            return {
                "user_id": user_id,
                "name": name,
                "message": "Account created successfully!"
            }
    
    def save_chat_message(self, user_id: str, session_id: str, message: str, response: str, tools_used: List[str] = None):
        """Save a chat message to the database"""
        chat_entry = {
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
            "response": response,
            "tools_used": tools_used or [],
            "timestamp": datetime.now().isoformat()
        }
        self.chats_db.insert(chat_entry)
    
    def get_user_chats(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all chats for a user grouped by session"""
        user_chats = self.chats_db.search(self.Chat.user_id == user_id)
        
        # Group by session_id
        sessions = {}
        for chat in user_chats:
            session_id = chat['session_id']
            if session_id not in sessions:
                sessions[session_id] = {
                    "session_id": session_id,
                    "messages": [],
                    "created_at": chat['timestamp'],
                    "updated_at": chat['timestamp']
                }
            
            sessions[session_id]["messages"].append({
                "message": chat["message"],
                "response": chat["response"],
                "tools_used": chat["tools_used"],
                "timestamp": chat["timestamp"]
            })
            
            # Update last message time
            if chat['timestamp'] > sessions[session_id]["updated_at"]:
                sessions[session_id]["updated_at"] = chat['timestamp']
        
        # Convert to list and sort by updated_at desc
        chats_list = list(sessions.values())
        chats_list.sort(key=lambda x: x["updated_at"], reverse=True)
        
        return chats_list
    
    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all messages in a specific session"""
        messages = self.chats_db.search(self.Chat.session_id == session_id)
        messages.sort(key=lambda x: x["timestamp"])
        return messages

db_manager = DatabaseManager()

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
            # Create OpenAI LLM
            self.llm = ChatOpenAI(
                api_key=config.OPENAI_API_KEY,
                model=config.MODEL_NAME,
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
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming using LangGraph with OpenAI."""
        try:
            # Get session history
            history = session_manager.get_session(session_id)
            
            # Create streaming LLM for final response
            streaming_llm = ChatOpenAI(
                api_key=config.OPENAI_API_KEY,
                model=config.MODEL_NAME,
                streaming=True,
                max_retries=config.MAX_RETRIES,
                request_timeout=config.TIMEOUT,
            )
            
            # Create initial state
            initial_state = {
                "messages": history + [HumanMessage(content=message)]
            }
            
            # Track what we've sent
            tools_used = []
            final_messages = []
            all_content = ""
            
            # Use astream for processing with tools
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
                            
                            # Collect content for streaming
                            if hasattr(msg, 'content') and msg.content:
                                all_content = msg.content
                    
                    elif node_name == "tools":
                        # Tool was executed
                        messages = node_output.get("messages", [])
                        for msg in messages:
                            final_messages.append(msg)
                            if isinstance(msg, ToolMessage):
                                yield f"data: {json.dumps({'type': 'tool_end', 'output': str(msg.content)[:100]})}\n\n"
            
            # If we have tools used, re-generate final response with streaming
            if tools_used and all_content:
                # Stream the final response
                words = all_content.split(' ')
                for i, word in enumerate(words):
                    if i > 0:
                        yield f"data: {json.dumps({'type': 'content', 'data': ' '})}\n\n"
                    yield f"data: {json.dumps({'type': 'content', 'data': word})}\n\n"
                    await asyncio.sleep(config.STREAM_DELAY)
            elif not tools_used:
                # No tools used, stream directly from OpenAI
                async for chunk in streaming_llm.astream([HumanMessage(content=message)]):
                    if hasattr(chunk, 'content') and chunk.content:
                        yield f"data: {json.dumps({'type': 'content', 'data': chunk.content})}\n\n"
                        await asyncio.sleep(config.STREAM_DELAY)
                        all_content += chunk.content
                
                # Update final messages for session
                final_messages = [HumanMessage(content=message), AIMessage(content=all_content)]
            
            # Update session with all messages
            all_messages = history + final_messages
            session_manager.update_session(session_id, all_messages)
            
            # Save to database if user_id provided
            if user_id and all_content:
                try:
                    db_manager.save_chat_message(user_id, session_id, message, all_content, tools_used)
                except Exception as e:
                    logger.error(f"Failed to save chat to database: {e}")
            
            # Send completion
            yield f"data: {json.dumps({'type': 'done', 'tools_used': tools_used})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}\n{traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    async def process_streaming_direct_openai(
        self, 
        message: str, 
        session_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Direct OpenAI streaming without LangGraph - no tools."""
        try:
            # Create streaming LLM
            streaming_llm = ChatOpenAI(
                api_key=config.OPENAI_API_KEY,
                model=config.MODEL_NAME,
                streaming=True,
                max_retries=config.MAX_RETRIES,
                request_timeout=config.TIMEOUT,
            )
            
            # Get session history if exists
            messages = []
            if session_id:
                history = session_manager.get_session(session_id)
                messages = history.copy()
            
            # Add current message
            messages.append(HumanMessage(content=message))
            
            full_response = ""
            
            # Stream from OpenAI
            async for chunk in streaming_llm.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    content = chunk.content
                    full_response += content
                    # Send content chunk to client
                    yield f"data: {json.dumps({'type': 'content', 'data': content})}\n\n"
            
            # Update session if session_id provided
            if session_id and full_response:
                history = session_manager.get_session(session_id)
                history.append(HumanMessage(content=message))
                history.append(AIMessage(content=full_response))
                session_manager.update_session(session_id, history)
            
            # Send completion
            yield f"data: {json.dumps({'type': 'done', 'tools_used': []})}\n\n"
                    
        except Exception as e:
            logger.error(f"Direct streaming error: {str(e)}\n{traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    async def process(
        self, 
        message: str, 
        session_id: str, 
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
    title="LangGraph OpenAI Streaming API",
    description="FastAPI application for LangGraph with OpenAI o3-mini and streaming support",
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
            "api_configured": bool(config.OPENAI_API_KEY)
        }
    }

# Main streaming endpoint with LangGraph
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat responses with LangGraph and OpenAI (supports tools)."""
    try:
        if not request.session_id:
            import uuid
            request.session_id = str(uuid.uuid4())
        
        logger.info(f"Streaming chat request - Session: {request.session_id}, Message: {request.message[:50]}...")
        
        return StreamingResponse(
            agent.process_streaming_with_langgraph(
                request.message,
                request.session_id,
                request.user_id,
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

# Direct OpenAI streaming (no tools)
@app.post("/chat/direct-stream")
async def direct_stream(request: ChatRequest):
    """Direct OpenAI streaming without LangGraph - no tools."""
    try:
        if not request.session_id:
            import uuid
            request.session_id = str(uuid.uuid4())
        
        logger.info(f"Direct streaming request - Message: {request.message[:50]}...")
        
        return StreamingResponse(
            agent.process_streaming_direct_openai(
                request.message,
                request.session_id,
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

# Test OpenAI streaming endpoint
@app.post("/test/openai-stream")
async def test_openai_stream(request: ChatRequest):
    """Test OpenAI streaming to verify it works."""
    try:
        async def generate():
            try:
                streaming_llm = ChatOpenAI(
                    api_key=config.OPENAI_API_KEY,
                    model=config.MODEL_NAME,
                    streaming=True,
                )
                
                async for chunk in streaming_llm.astream([HumanMessage(content=request.message)]):
                    if hasattr(chunk, 'content') and chunk.content:
                        yield f"data: {json.dumps({'content': chunk.content})}\n\n"
                
                yield f"data: {json.dumps({'done': True})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )
        
    except Exception as e:
        logger.error(f"Test OpenAI streaming error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Authentication endpoint
@app.post("/auth", response_model=AuthResponse)
async def authenticate(request: AuthRequest):
    """Simple name-based authentication"""
    try:
        result = db_manager.authenticate_user(request.name)
        logger.info(f"Authentication for user: {request.name}")
        
        return AuthResponse(
            user_id=result["user_id"],
            name=result["name"],
            message=result["message"]
        )
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication error: {str(e)}"
        )

# User history endpoint
@app.get("/history/{user_id}", response_model=UserHistoryResponse)
async def get_user_history(user_id: str):
    """Get all chat history for a user"""
    try:
        chats = db_manager.get_user_chats(user_id)
        logger.info(f"Retrieved history for user: {user_id}")
        
        return UserHistoryResponse(
            user_id=user_id,
            chats=chats
        )
    except Exception as e:
        logger.error(f"History retrieval error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"History error: {str(e)}"
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
        app,
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
        access_log=True,
        workers=1
    )