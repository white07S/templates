"""FastAPI main application with /chat endpoint."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
from typing import Dict, Any
from datetime import datetime
import uuid
from openai import AsyncOpenAI
import re
import json

from config import settings
from models import (
    ChatRequest, ChatResponse, Message, MessageRole,
    PerformanceMetrics
)
from memory_manager import MemoryOrchestrator
from storage import ConversationStore, MemoryStore
from tools import executor as tool_executor
from react_agent import ReActAgent

# Global instances
memory_orchestrator = MemoryOrchestrator()
conversation_store = ConversationStore()
memory_store = MemoryStore()
react_agent = ReActAgent() if settings.enable_react_agent else None

import nltk
nltk.download('punkt')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    print("Starting Chat Backend with Dual-Level Memory...")
    print(f"Using model: {settings.chat_model}")
    print(f"Data directory: {settings.data_dir}")

    yield

    # Shutdown
    print("Shutting down...")
    conversation_store.close()
    memory_store.close()

# Create FastAPI app
app = FastAPI(
    title="Chat Backend with Dual-Level Memory",
    description="Advanced chat system with session and user-level memory using GPT-4o",
    version="1.0.0",
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

class ChatProcessor:
    """Core chat processing logic with intelligent routing and context awareness."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.performance_metrics = {}
        # Track tool results in recent context for follow-up detection
        self.recent_tool_results = {}  # session_id -> list of recent tool results

    def _extract_recent_tool_results(self, context: list) -> Dict[str, Any]:
        """Extract recent tool execution results from context."""
        tool_results = {}

        # Look through context for tool results (typically in the last few messages)
        for i, msg in enumerate(context):
            if isinstance(msg.get("content"), str):
                # Check if this looks like a tool result (JSON format)
                content = msg.get("content", "")
                if content.startswith("{") and content.endswith("}"):
                    try:
                        parsed = json.loads(content)
                        # Store tool results with their type
                        if "result" in parsed or "memories" in parsed or "fact" in parsed:
                            tool_results.update(parsed)
                    except:
                        pass

        return tool_results

    def _is_follow_up_query(self, query: str, context: list) -> bool:
        """
        Detect if the query is a follow-up to previous context.
        Uses linguistic patterns and context analysis.
        """
        query_lower = query.lower().strip()

        # Patterns that indicate follow-up questions
        follow_up_patterns = [
            # Pronouns and references
            r'^(what|how|why|when|where|who)\s+(is|are|was|were)?\s*(it|that|this|these|those|they)',
            r'^(can|could|would|should)\s+(you|i|we|it|that)',
            r'^(tell|show|explain|give)\s+me\s+more',
            r'^more\s+(about|on|regarding|details)',

            # Direct references to previous content
            r'^(and|also|but|however|furthermore|additionally)',
            r'^(based on|given|considering|regarding)\s+(that|this|the above)',
            r'^what about',
            r'^how about',
            r'^in (that|this) case',

            # Clarification patterns
            r'^(so|then|therefore)',
            r'^does (that|this|it) mean',
            r'^are you saying',
            r'^in other words',
            r'^to clarify',

            # Short follow-ups
            r'^why\?*$',
            r'^how\?*$',
            r'^what\?*$',
            r'^explain\.*$',
            r'^more\.*$',
            r'^continue\.*$',
            r'^go on\.*$',

            # References to "the" something (implying previous mention)
            r'^(what|how|why|when|where)\s+.*\bthe\b',
            r'^(using|with|from)\s+the\s+(result|data|information|memory|fact)',
        ]

        # Check for follow-up patterns
        for pattern in follow_up_patterns:
            if re.search(pattern, query_lower):
                return True

        # Check for very short queries (often follow-ups)
        if len(query.split()) <= 3 and any(word in query_lower for word in ["it", "that", "this", "what", "why", "how"]):
            return True

        # Check if query references specific data from recent context
        if context and len(context) > 1:
            # Get last assistant message
            for msg in reversed(context):
                if msg.get("role") == "assistant":
                    last_response = msg.get("content", "").lower()
                    # Check if query asks about specific terms from last response
                    response_keywords = set(re.findall(r'\b\w{4,}\b', last_response))
                    query_words = set(re.findall(r'\b\w{4,}\b', query_lower))

                    # High overlap suggests follow-up
                    if response_keywords and query_words:
                        overlap = len(response_keywords & query_words) / len(query_words)
                        if overlap > 0.3:  # 30% of query words were in previous response
                            return True
                    break

        return False

    def _requires_tool_execution(self, query: str, context: list) -> bool:
        """
        Determine if the query actually requires tool execution.
        This is different from complexity - even simple queries might need tools.
        """
        query_lower = query.lower()

        # Direct tool indicators
        tool_required_patterns = [
            # Time/date requests
            r'\b(current|what|tell me the)\s+(time|date|day|year)',
            r'\bwhat\s+time\s+is\s+it',

            # Calculation requests
            r'\b(calculate|compute|solve|evaluate)\b',
            r'\b(sum|add|subtract|multiply|divide)\b',
            r'[0-9]+\s*[\+\-\*/]\s*[0-9]+',  # Mathematical expressions

            # Memory operations
            r'\b(save|store|remember|recall|search)\s+(my|the)?\s*(memory|memories|preference)',
            r'\b(what|find|get)\s+.*\s+memories?\b',

            # Unit conversions
            r'\b(convert|change)\s+.*\s+(to|into)\b',
            r'\b(celsius|fahrenheit|meters|feet|pounds|kilograms)\b',

            # Fact generation
            r'\b(generate|give|tell)\s+.*\s+(random|interesting)?\s*fact',

            # Task/plan creation
            r'\b(create|make|generate)\s+.*\s+(task|plan|list)',

            # Text analysis
            r'\b(analyze|count|summarize)\s+.*\s+(text|words|this)',
        ]

        for pattern in tool_required_patterns:
            if re.search(pattern, query_lower):
                return True

        # Check for explicit tool names
        tool_names = ["calculate", "convert", "time", "memory", "save", "search", "generate", "create", "analyze"]
        if any(tool in query_lower for tool in tool_names):
            return True

        return False

    def _can_answer_from_context(self, query: str, context: list, recent_tool_results: Dict) -> bool:
        """
        Check if the query can be answered from existing context without new tool calls.
        """
        if not recent_tool_results and not context:
            return False

        query_lower = query.lower()

        # Questions about previous results
        context_answerable_patterns = [
            r'^what (did|does) (that|it|the result) (mean|show|indicate)',
            r'^explain (that|the|this) (result|output|response)',
            r'^why (did|does|is) (it|that|the)',
            r'^(is|are) (that|this|it|they) (correct|right|accurate)',
            r'^what (can|could|should) (i|we) (do|learn|understand) (from|with) (this|that|it)',
            r'^(so|then) what',
            r'^based on (this|that|the above)',
        ]

        for pattern in context_answerable_patterns:
            if re.search(pattern, query_lower):
                return True

        # Check if asking for clarification about tool results
        if recent_tool_results:
            # Questions about specific values in results
            for key, value in recent_tool_results.items():
                if key.lower() in query_lower or str(value).lower() in query_lower:
                    # Asking about something that's already in the results
                    if any(q in query_lower for q in ["what", "why", "how", "explain", "mean"]):
                        return True

        return False

    def _should_use_react_agent(self, query: str, context: list) -> bool:
        """
        Intelligent decision making for when to use ReAct agent.
        Based on research and best practices for agent routing.
        """
        if not react_agent or not settings.enable_react_agent:
            return False

        # Extract recent tool results for context awareness
        recent_tool_results = self._extract_recent_tool_results(context)

        # Step 1: Check if this is a follow-up that can be answered from context
        if self._is_follow_up_query(query, context):
            # Follow-up detected - check if we can answer from context
            if self._can_answer_from_context(query, context, recent_tool_results):
                print(f"📝 Follow-up query detected that can be answered from context: {query[:50]}...")
                return False

        # Step 2: Check if tools are actually required
        needs_tools = self._requires_tool_execution(query, context)

        # Step 3: Analyze query complexity for multi-step operations
        query_lower = query.lower()

        # Multi-step indicators (requires orchestration)
        multi_step_patterns = [
            r'\b(first|then|next|after|finally)\b.*\b(then|next|after|finally)\b',  # Explicit sequencing
            r'\b(and then|after that|following this)\b',  # Sequential connectors
            r'\b(step by step|one by one|in order)\b',  # Process indicators
            r'\b(compare|analyze)\b.*\b(and|then|with)\b.*\b(calculate|convert|find)\b',  # Multiple operations
        ]

        has_multi_steps = any(re.search(pattern, query_lower) for pattern in multi_step_patterns)

        # Count distinct operations requested
        operations = {
            "calculate": bool(re.search(r'\b(calculate|compute|solve)\b', query_lower)),
            "convert": bool(re.search(r'\b(convert|change.*to)\b', query_lower)),
            "memory": bool(re.search(r'\b(save|remember|recall|search.*memory)\b', query_lower)),
            "analyze": bool(re.search(r'\b(analyze|compare|investigate)\b', query_lower)),
            "time": bool(re.search(r'\b(current time|what time|date)\b', query_lower)),
            "create": bool(re.search(r'\b(create|make|generate)\b', query_lower)),
        }

        operation_count = sum(operations.values())

        # Step 4: Make routing decision based on comprehensive analysis
        use_react = False
        reason = ""

        if not needs_tools:
            # No tools needed at all
            use_react = False
            reason = "No tool execution required"
        elif has_multi_steps or operation_count >= 2:
            # Multiple steps or operations - use ReAct for orchestration
            use_react = True
            reason = f"Multi-step operation detected ({operation_count} operations)"
        elif needs_tools and len(query.split()) > 20:
            # Complex query with tools needed
            use_react = True
            reason = "Complex query requiring tool orchestration"
        else:
            # Simple tool use - regular function calling is sufficient
            use_react = False
            reason = "Simple tool execution - using standard function calling"

        if use_react:
            print(f"🤖 ReAct Agent selected: {reason}")
        elif needs_tools:
            print(f"🔧 Standard tool execution selected: {reason}")
        else:
            print(f"💬 Direct response selected: {reason}")

        return use_react

    async def process_chat_request(
        self,
        request: ChatRequest,
        background_tasks: BackgroundTasks
    ) -> ChatResponse:
        """Process a chat request with memory integration."""
        start_time = datetime.utcnow()

        try:
            # Get memory instances
            session_memory = memory_orchestrator.get_session_memory(
                request.session_id, request.user_id
            )
            user_memory = memory_orchestrator.get_user_memory(request.user_id)

            # Calculate token budget
            token_budget = memory_orchestrator.calculate_token_budget()

            # Add user message to session
            user_message = Message(
                role=MessageRole.USER,
                content=request.message
            )
            await session_memory.add_message(user_message)

            # Get recent conversation context
            recent_messages, context_tokens = await session_memory.get_recent_context(
                token_budget.conversation_tokens
            )

            # Get relevant user memories
            relevant_memories, memory_tokens = await user_memory.get_relevant_memories(
                request.message,
                token_budget.memory_tokens
            )

            # Build context for LLM
            context = await self._build_context(
                recent_messages,
                relevant_memories,
                request
            )

            # Determine if we should use ReAct agent
            use_react = self._should_use_react_agent(request.message, context)
            react_metadata = {}
            tool_results_for_session = {}

            if use_react:
                print(f"🤖 Using ReAct agent for complex query: {request.message[:50]}...")
                # Use ReAct agent for complex queries
                response_content, react_metadata = await react_agent.process_query(
                    request.message,
                    request.user_id,
                    context,
                    check_tools_needed=True  # New parameter to check if tools are needed
                )
                # Store any tool results from ReAct execution
                if "tool_results" in react_metadata:
                    tool_results_for_session = react_metadata["tool_results"]
            else:
                # Use traditional function calling for simpler queries
                response_content, tool_results_for_session = await self._generate_response_with_tracking(
                    context, request.user_id
                )

            # Store tool results for follow-up detection
            if tool_results_for_session:
                self.recent_tool_results[request.session_id] = tool_results_for_session

            # Add assistant message to session
            assistant_message = Message(
                role=MessageRole.ASSISTANT,
                content=response_content
            )
            await session_memory.add_message(assistant_message)

            # Schedule background memory processing
            background_tasks.add_task(
                self._process_memories_async,
                request.session_id,
                request.user_id
            )

            # Track performance
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Prepare metadata
            metadata = {
                "processing_time_ms": processing_time,
                "context_tokens": context_tokens,
                "memory_tokens": memory_tokens,
                "memories_count": len(relevant_memories),
                "model": settings.chat_model,
                "used_react_agent": use_react
            }

            # Add ReAct metadata if applicable
            if react_metadata:
                metadata.update(react_metadata)

            return ChatResponse(
                session_id=request.session_id,
                user_id=request.user_id,
                response=response_content,
                memories_used=[mem.memory_id for mem in relevant_memories],
                metadata=metadata
            )

        except Exception as e:
            print(f"Error processing chat request: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _build_context(
        self,
        recent_messages: list,
        relevant_memories: list,
        request: ChatRequest
    ) -> list:
        """Build the context for LLM generation."""
        context = []

        # System prompt
        system_prompt = self._build_system_prompt(relevant_memories, request.user_id)
        context.append({
            "role": "system",
            "content": system_prompt
        })

        # Add memory context if available
        if relevant_memories:
            memory_context = "Relevant memories:\n"
            for i, memory in enumerate(relevant_memories, 1):
                content = memory.compressed_content or memory.content
                memory_context += f"{i}. {content}\n"

            context.append({
                "role": "system",
                "content": memory_context
            })

        # Add conversation history
        for message in recent_messages:
            context.append({
                "role": message.role,
                "content": message.content
            })

        return context

    def _build_system_prompt(self, relevant_memories: list, user_id: str) -> str:
        """Build the system prompt with memory context."""
        base_prompt = """You are an advanced AI assistant with access to user memories and conversation history.

        Key capabilities:
        - Access to user's past conversations and learned preferences
        - Memory of facts, preferences, and context from previous sessions
        - Ability to reference and build upon previous interactions

        Guidelines:
        - Use memories naturally in conversation when relevant
        - Don't explicitly mention "I remember" unless contextually appropriate
        - Personalize responses based on user's preferences and history
        - Maintain consistency with previous interactions
        - Be helpful, informative, and conversational
        """

        if relevant_memories:
            base_prompt += f"\nYou have access to {len(relevant_memories)} relevant memories for this conversation."

        return base_prompt

    async def _generate_response(self, context: list, user_id: str) -> str:
        """Generate response using OpenAI API with function calling support."""
        response_content, _ = await self._generate_response_with_tracking(context, user_id)
        return response_content

    async def _generate_response_with_tracking(self, context: list, user_id: str) -> tuple[str, Dict]:
        """Generate response with tool result tracking for context awareness."""
        try:
            tool_execution_results = {}

            # Get available tools
            available_tools = tool_executor.get_available_tools()

            # Make initial API call with tools
            response = await self.client.chat.completions.create(
                model=settings.chat_model,
                messages=context,
                tools=available_tools,
                tool_choice="auto",
                max_tokens=1000,
                temperature=0.7,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )

            response_message = response.choices[0].message

            # Check if the model wants to call tools
            if response_message.tool_calls:
                print(f"🔧 Model wants to call {len(response_message.tool_calls)} tools")

                # Add the assistant's message to context
                context.append({
                    "role": "assistant",
                    "content": response_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in response_message.tool_calls
                    ]
                })

                # Execute tools
                tool_results = await tool_executor.process_tool_calls([
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in response_message.tool_calls
                ], user_id=user_id)

                # Store tool results for tracking
                for i, result in enumerate(tool_results):
                    if "content" in result:
                        try:
                            parsed_result = json.loads(result["content"])
                            tool_name = response_message.tool_calls[i].function.name
                            tool_execution_results[tool_name] = parsed_result
                        except:
                            pass

                # Add tool results to context
                for result in tool_results:
                    context.append(result)

                # Make second API call with tool results
                final_response = await self.client.chat.completions.create(
                    model=settings.chat_model,
                    messages=context,
                    max_tokens=1000,
                    temperature=0.7,
                    presence_penalty=0.1,
                    frequency_penalty=0.1
                )

                return final_response.choices[0].message.content, tool_execution_results

            else:
                # No tools called, return direct response
                return response_message.content or "I understand, but I don't have a specific response right now.", {}

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again.", {}

    async def _process_memories_async(self, session_id: str, user_id: str):
        """Background task for memory processing."""
        try:
            await memory_orchestrator.process_conversation_for_memories(
                session_id, user_id
            )
            print(f"✓ Processed memories for session {session_id}")
        except Exception as e:
            print(f"✗ Error processing memories: {e}")
            import traceback
            traceback.print_exc()

# Create chat processor
chat_processor = ChatProcessor()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks
) -> ChatResponse:
    """
    Main chat endpoint with dual-level memory.

    Takes session-id and user-id as payload along with the message.
    Returns AI response with memory context and metadata.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    return await chat_processor.process_chat_request(request, background_tasks)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model": settings.chat_model,
        "version": "1.0.0"
    }

@app.get("/memory/stats/{user_id}")
async def get_memory_stats(user_id: str):
    """Get memory statistics for a user."""
    try:
        user_memory = memory_orchestrator.get_user_memory(user_id)
        user_memories = await user_memory.memory_store.get_user_memories(user_id)

        stats = {
            "user_id": user_id,
            "total_memories": len(user_memories),
            "memory_types": {},
            "avg_importance": 0.0,
            "recent_memories": 0
        }

        if user_memories:
            # Count by type
            for memory in user_memories:
                mem_type = memory.memory_type
                stats["memory_types"][mem_type] = stats["memory_types"].get(mem_type, 0) + 1

            # Calculate average importance
            stats["avg_importance"] = sum(m.importance_score for m in user_memories) / len(user_memories)

            # Count recent memories (last 7 days)
            week_ago = datetime.utcnow().timestamp() - (7 * 24 * 3600)
            stats["recent_memories"] = sum(
                1 for m in user_memories
                if m.created_at.timestamp() > week_ago
            )

        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/history/{session_id}")
async def get_session_history(session_id: str, user_id: str):
    """Get conversation history for a session."""
    try:
        conversation = await conversation_store.get_conversation(session_id, user_id)

        if not conversation:
            return {
                "session_id": session_id,
                "user_id": user_id,
                "messages": [],
                "message_count": 0
            }

        return {
            "session_id": session_id,
            "user_id": user_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in conversation.messages
            ],
            "message_count": len(conversation.messages),
            "created_at": conversation.timestamp.isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}")
async def delete_session(session_id: str, user_id: str):
    """Delete a session and its conversation history."""
    try:
        # Remove from memory orchestrator
        if session_id in memory_orchestrator.session_memories:
            del memory_orchestrator.session_memories[session_id]

        # Note: In production, you'd also delete from TinyDB
        # For now, we'll just clear the in-memory reference

        return {
            "message": f"Session {session_id} deleted successfully",
            "session_id": session_id,
            "user_id": user_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug/process-memories/{session_id}")
async def debug_process_memories(session_id: str, user_id: str):
    """Debug endpoint to manually trigger memory processing."""
    try:
        await memory_orchestrator.process_conversation_for_memories(session_id, user_id)
        return {"message": "Memory processing completed", "session_id": session_id, "user_id": user_id}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/session-state/{session_id}")
async def debug_session_state(session_id: str, user_id: str):
    """Debug endpoint to inspect session state."""
    try:
        session_memory = memory_orchestrator.get_session_memory(session_id, user_id)
        await session_memory._load_existing_conversation()

        return {
            "session_id": session_id,
            "user_id": user_id,
            "conversation_buffer_size": len(session_memory.conversation_buffer),
            "context_summary": session_memory.context_summary,
            "loaded": session_memory._loaded,
            "recent_messages": [
                {
                    "role": msg.role,
                    "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in list(session_memory.conversation_buffer)[-5:]
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools")
async def get_available_tools():
    """Get list of available function calling tools."""
    try:
        tools = tool_executor.get_available_tools()
        return {
            "tools": tools,
            "count": len(tools),
            "tool_names": [tool["function"]["name"] for tool in tools]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/execute")
async def execute_tool_directly(tool_name: str, arguments: dict = None):
    """Directly execute a tool for testing purposes."""
    try:
        if arguments is None:
            arguments = {}

        result = await tool_executor.registry.execute_tool(tool_name, arguments)
        return {
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug/react")
async def debug_react_agent(query: str, user_id: str = "test_user"):
    """Test the ReAct agent directly with a complex query."""
    try:
        if not react_agent:
            raise HTTPException(status_code=400, detail="ReAct agent is disabled")

        # Build minimal context
        context = [
            {"role": "system", "content": "You are testing the ReAct agent."},
            {"role": "user", "content": query}
        ]

        response, metadata = await react_agent.process_query(query, user_id, context)

        return {
            "query": query,
            "user_id": user_id,
            "response": response,
            "metadata": metadata,
            "status": "completed"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config/react")
async def get_react_config():
    """Get ReAct agent configuration."""
    return {
        "enabled": settings.enable_react_agent,
        "agent_available": react_agent is not None,
        "complexity_threshold": settings.react_complexity_threshold,
        "max_actions": settings.react_max_actions,
        "parallel_limit": settings.react_parallel_limit
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        workers=1  # Single worker for development
    )
