"""FastAPI main application with /chat endpoint."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
from typing import Dict, Any
from datetime import datetime
import uuid
from openai import AsyncOpenAI

from config import settings
from models import (
    ChatRequest, ChatResponse, Message, MessageRole,
    PerformanceMetrics
)
from memory_manager import MemoryOrchestrator
from storage import ConversationStore, MemoryStore
from tools import executor as tool_executor
from react_agent import ReActAgent
from optimized_openai import OptimizedChatClient

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
    print("\n🛑 Shutting down gracefully...")

    # Close FAISS vector stores cleanly
    try:
        from embeddings import VectorMemoryStore
        vector_store = VectorMemoryStore()
        if hasattr(vector_store.vector_store, 'cleanup'):
            vector_store.vector_store.cleanup()
    except Exception as e:
        print(f"Warning: Could not cleanup FAISS: {e}")

    # Close database connections
    conversation_store.close()
    memory_store.close()

    print("✅ Shutdown complete")

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
    """Core chat processing logic with performance optimizations."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.optimized_client = OptimizedChatClient()
        self.performance_metrics = {}

        # Performance tracking
        self._request_count = 0
        self._total_processing_time = 0.0
        self._cache_hits = 0

    def _should_use_react_agent(self, query: str, context: list) -> bool:
        """Determine if a query should use the ReAct agent."""
        if not react_agent or not settings.enable_react_agent:
            return False

        # Enhanced heuristics for ReAct agent usage
        complexity_indicators = [
            # Multiple tasks connectors
            " and ", " then ", " also ", " plus ", " both ", " after ", " before ",
            # Research/analysis tasks
            "analyze", "compare", "research", "investigate", "find out", "differences",
            "similarities", "pros and cons", "advantages", "disadvantages",
            # Sequential operations
            "first", "after that", "next", "finally", "step by step", "in order",
            # Complex questions
            "how can i", "what are the steps", "help me plan", "walk me through",
            # Multiple tools likely needed
            "calculate and", "time and", "remember and", "convert and", "then calculate",
            "then tell", "then analyze", "then find", "and calculate", "and tell",
            # Time-based sequential tasks
            "until", "between", "from", "to"
        ]

        query_lower = query.lower()
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in query_lower)

        # Additional checks for complexity
        # Check for multiple verbs indicating multiple actions
        action_verbs = ["calculate", "analyze", "compare", "find", "tell", "convert", "determine", "help"]
        verb_count = sum(1 for verb in action_verbs if verb in query_lower)

        # Use ReAct if:
        # 1. High complexity score (2+ indicators)
        # 2. Multiple action verbs (2+ verbs)
        # 3. Very long query (>15 words)
        return complexity_score >= 2 or verb_count >= 2 or len(query.split()) > 15

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

            # Get recent conversation context and user memories in parallel
            context_task = session_memory.get_recent_context(token_budget.conversation_tokens)
            memory_task = user_memory.get_relevant_memories(request.message, token_budget.memory_tokens)

            (recent_messages, context_tokens), (relevant_memories, memory_tokens) = await asyncio.gather(
                context_task, memory_task
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

            if use_react:
                print(f"🤖 Using ReAct agent for complex query: {request.message[:50]}...")
                # Use ReAct agent for complex queries
                response_content, react_metadata = await react_agent.process_query(
                    request.message,
                    request.user_id,
                    context
                )
            else:
                # Use traditional function calling for simpler queries
                response_content = await self._generate_response(context, request.user_id)

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
            self._request_count += 1
            self._total_processing_time += processing_time

            # Prepare metadata with performance insights
            metadata = {
                "processing_time_ms": processing_time,
                "context_tokens": context_tokens,
                "memory_tokens": memory_tokens,
                "memories_count": len(relevant_memories),
                "model": settings.chat_model,
                "used_react_agent": use_react,
                "avg_processing_time": self._total_processing_time / self._request_count,
                "request_count": self._request_count
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
        """Generate response using optimized OpenAI API with function calling support."""
        try:
            # Get available tools
            available_tools = tool_executor.get_available_tools()

            # Make initial API call with tools using optimized client
            response = await self.optimized_client.create_completion(
                messages=context,
                tools=available_tools,
                tool_choice="auto",
                max_tokens=1000,
                temperature=0.7,
                presence_penalty=0.1,
                frequency_penalty=0.1,
                use_cache=True  # Enable caching for similar requests
            )

            response_message = response.choices[0].message

            # Check if the model wants to call tools (Two-step workflow)
            if response_message.tool_calls:
                print(f"🔧 Model wants to call {len(response_message.tool_calls)} tools")

                # Step 1: Add the assistant's initial response with tool calls to context
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

                # Step 2: Execute tools and get results
                tool_results = await tool_executor.process_tool_calls_for_llm(response_message.tool_calls)

                # Step 3: Add tool results to context
                for result in tool_results:
                    context.append(result)

                # Step 4: Make second API call with tool results for final response
                print("🔄 Making second LLM call with tool results...")
                final_response = await self.optimized_client.create_completion(
                    messages=context,
                    max_tokens=1000,
                    temperature=0.7,
                    presence_penalty=0.1,
                    frequency_penalty=0.1,
                    use_cache=False  # Don't cache final responses as they include tool results
                )

                print("✅ Two-step tool workflow completed")
                return final_response.choices[0].message.content

            else:
                # No tools called, return direct response
                return response_message.content or "I understand, but I don't have a specific response right now."

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again."

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

@app.get("/tools/schema")
async def get_tools_schema():
    """Get the JSON schema for all available tools."""
    try:
        tools = tool_executor.get_available_tools()
        return {
            "tools": tools,
            "count": len(tools),
            "format": "json_schema",
            "tool_functions": list(tool_executor.registry.tool_functions.keys())
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

@app.get("/performance/stats")
async def get_performance_stats():
    """Get comprehensive performance statistics."""
    try:
        # Get stats from all optimized components
        storage_stats = conversation_store.get_cache_stats()
        memory_storage_stats = memory_store.get_cache_stats()
        openai_stats = chat_processor.optimized_client.get_stats()
        tool_stats = tool_executor.get_performance_stats()

        # Get FAISS stats if available
        faiss_stats = {}
        try:
            from embeddings import VectorMemoryStore
            vector_store = VectorMemoryStore()
            faiss_stats = vector_store.vector_store.get_stats()
        except Exception as e:
            faiss_stats = {"error": str(e)}

        return {
            "overall": {
                "total_requests": chat_processor._request_count,
                "avg_processing_time_ms": chat_processor._total_processing_time / max(chat_processor._request_count, 1),
                "cache_hits": chat_processor._cache_hits
            },
            "storage": {
                "conversation_store": storage_stats,
                "memory_store": memory_storage_stats
            },
            "openai": openai_stats,
            "tools": tool_stats,
            "faiss": faiss_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/performance/clear-caches")
async def clear_all_caches():
    """Clear all caches for testing purposes."""
    try:
        # Clear all caches
        conversation_store.query_cache.clear()
        conversation_store.document_cache.clear()
        memory_store.query_cache.clear()
        memory_store.document_cache.clear()

        chat_processor.optimized_client.response_cache.clear()
        chat_processor.optimized_client.compression_cache.clear()

        tool_executor.registry.clear_cache()

        # Clear FAISS search cache if available
        try:
            from embeddings import VectorMemoryStore
            vector_store = VectorMemoryStore()
            vector_store.vector_store.search_cache.clear()
        except Exception:
            pass

        return {
            "message": "All caches cleared successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        workers=1  # Single worker for development
    )
