import os
from typing import Literal, Dict, List, AsyncGenerator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
import math
import json
from datetime import datetime
import uuid

# Configure your vLLM endpoint
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
MODEL_NAME = os.getenv("MODEL_NAME", "your-model-name")

# Define simple tools for testing
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    weather_data = {
        "new york": "Sunny, 72°F",
        "london": "Cloudy, 59°F", 
        "tokyo": "Rainy, 65°F",
        "paris": "Partly cloudy, 68°F"
    }
    return weather_data.get(location.lower(), f"Weather data not available for {location}")

@tool
def calculate(expression: str) -> str:
    """Perform basic mathematical calculations. 
    Input should be a valid Python mathematical expression."""
    try:
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    search_results = {
        "langchain": "LangChain is a framework for developing applications powered by language models.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        "vllm": "vLLM is a fast and easy-to-use library for LLM inference and serving."
    }
    
    for key, value in search_results.items():
        if key in query.lower():
            return value
    
    return f"Search results for '{query}': No specific results found in our mock database."

# Create the tools list
tools = [get_weather, calculate, search_web]

# Session memory storage - stores conversation history per session
sessions: Dict[str, List[BaseMessage]] = {}

class LangGraphAgent:
    def __init__(self):
        # Initialize the LLM client
        self.llm = ChatOpenAI(
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            model=MODEL_NAME,
            temperature=0.7,
            streaming=True,
        )
        
        # Bind tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(tools)
        
        # Build the graph
        self.app = self._build_graph()
        self.sessions = sessions
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        # Define the graph state
        class AgentState(MessagesState):
            """State of the agent."""
            pass
        
        # Define the function that calls the model
        async def call_model(state: AgentState):
            messages = state["messages"]
            response = await self.llm_with_tools.ainvoke(messages)
            return {"messages": [response]}
        
        # Define the function that determines whether to continue or not
        def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
            messages = state["messages"]
            last_message = messages[-1]
            
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "__end__"
        
        # Build the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(tools))
        
        # Set the entrypoint
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "__end__": "__end__",
            }
        )
        
        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")
        
        # Compile and return
        return workflow.compile()
    
    def get_or_create_session(self, session_id: str) -> List[BaseMessage]:
        """Get or create a session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]
    
    def add_to_session(self, session_id: str, messages: List[BaseMessage]):
        """Add messages to session history"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].extend(messages)
        # Keep only last 20 messages to prevent context overflow
        if len(self.sessions[session_id]) > 20:
            self.sessions[session_id] = self.sessions[session_id][-20:]
    
    async def stream_response(self, message: str, session_id: str = None) -> AsyncGenerator[str, None]:
        """Stream response with session memory"""
        try:
            # Generate session ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Get session history
            session_messages = self.get_or_create_session(session_id)
            
            # Create user message
            user_message = HumanMessage(content=message)
            
            # Combine session history with new message
            all_messages = session_messages + [user_message]
            
            # Create initial state
            state = {"messages": all_messages}
            
            # Track new messages for session update
            new_messages = [user_message]
            
            # Stream the response
            current_content = ""
            async for event in self.app.astream_events(state, version="v2"):
                if event["event"] == "on_chat_model_stream":
                    if "chunk" in event["data"] and hasattr(event["data"]["chunk"], "content"):
                        content = event["data"]["chunk"].content
                        if content:
                            current_content += content
                            # Yield content chunks in the original format
                            yield f"data: {json.dumps({'content': content, 'type': 'content'})}\n\n"
                
                elif event["event"] == "on_chat_model_end":
                    # Get the full AI message
                    if "output" in event["data"]:
                        ai_message = event["data"]["output"]
                        new_messages.append(ai_message)
                
                elif event["event"] == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    tool_input = event["data"].get("input", {})
                    yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool_name, 'args': tool_input})}\n\n"
                
                elif event["event"] == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    tool_output = event["data"].get("output", "")
                    yield f"data: {json.dumps({'type': 'tool_result', 'tool': tool_name, 'result': tool_output})}\n\n"
                    
                    # Tool results will trigger another agent call, so continue streaming
            
            # Add all new messages to session
            self.add_to_session(session_id, new_messages)
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    def clear_session(self, session_id: str):
        """Clear session history"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def get_session_info(self, session_id: str) -> dict:
        """Get information about a session"""
        messages = self.get_or_create_session(session_id)
        return {
            "session_id": session_id,
            "message_count": len(messages),
            "messages": [
                {
                    "type": msg.__class__.__name__,
                    "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                }
                for msg in messages
            ]
        }