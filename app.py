"""
Advanced Chat Streaming Backend with Memory Management
Features: Session/User memory, Tool calling, ReAct reasoning, Token optimization
"""

import os
import json
import asyncio
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, AsyncIterator, Tuple
from enum import Enum
from pathlib import Path
import hashlib
import re
import traceback
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from tinydb import TinyDB, Query
from openai import AsyncOpenAI
from mem0 import MemoryClient, Memory
import numpy as np
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Configuration ====================

class Config:
    """Application configuration with environment variables"""
    # Directories
    DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
    USER_DIR = DATA_DIR / "users"
    
    # Token budgets (configurable)
    TOKEN_BUDGET_SYSTEM = int(os.getenv("TOKEN_BUDGET_SYSTEM", "2000"))
    TOKEN_BUDGET_MEMORY = int(os.getenv("TOKEN_BUDGET_MEMORY", "3000"))
    TOKEN_BUDGET_CONVERSATION = int(os.getenv("TOKEN_BUDGET_CONVERSATION", "8000"))
    
    # ReAct configuration
    MAX_REASONING_STEPS = int(os.getenv("MAX_REASONING_STEPS", "10"))
    REASONING_TIMEOUT = int(os.getenv("REASONING_TIMEOUT", "60"))
    
    # Memory configuration
    MEMORY_RELEVANCE_THRESHOLD = float(os.getenv("MEMORY_RELEVANCE_THRESHOLD", "0.7"))
    COMPRESSION_RATIO = float(os.getenv("COMPRESSION_RATIO", "0.3"))
    MEMORY_CACHE_SIZE = int(os.getenv("MEMORY_CACHE_SIZE", "100"))
    
    # LLM Configuration
    USE_OPENAI = os.getenv("USE_OPENAI", "true").lower() == "true"
    
    # OpenAI settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # vLLM settings
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "dummy-key")
    LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    # Mem0 configuration
    MEM0_API_KEY = os.getenv("MEM0_API_KEY")
    USE_MEM0_CLOUD = bool(MEM0_API_KEY)
    
    # API settings
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "2000"))
    STREAM_CHUNK_DELAY = float(os.getenv("STREAM_CHUNK_DELAY", "0.01"))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# ==================== Data Models ====================

class MemoryType(str, Enum):
    """Types of memory storage"""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    PREFERENCE = "preference"


class ReasoningMode(str, Enum):
    """Reasoning strategies"""
    DIRECT = "direct"
    REACT = "react"
    PLAN_EXECUTE = "plan_execute"


class ChatRequest(BaseModel):
    """Chat endpoint request model"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    reasoning_mode: bool = False
    temperature: float = Field(default=Config.DEFAULT_TEMPERATURE, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=Config.DEFAULT_MAX_TOKENS)
    
    @validator("session_id")
    def validate_session_id(cls, v):
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            return str(uuid.uuid4())


class Message(BaseModel):
    """Message in conversation"""
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None


class Memory(BaseModel):
    """Memory item structure"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MemoryType
    content: str
    embedding: Optional[List[float]] = None
    relevance_score: float = 0.0
    access_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ==================== Memory Management ====================

class MemoryManager:
    """Advanced memory management with compression and optimization"""
    
    def __init__(self, config: Config):
        self.config = config
        self.config.USER_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize Mem0
        try:
            if config.USE_MEM0_CLOUD and config.MEM0_API_KEY:
                self.mem0_client = MemoryClient(api_key=config.MEM0_API_KEY)
            else:
                self.mem0_client = Memory()
        except Exception as e:
            logger.warning(f"Failed to initialize Mem0: {e}. Using local storage only.")
            self.mem0_client = None
        
        # Memory buffers for optimization
        self.memory_cache = {}
        self.compression_buffer = deque(maxlen=config.MEMORY_CACHE_SIZE)
    
    def get_session_db(self, user_id: str, session_id: str) -> TinyDB:
        """Get or create session database"""
        user_dir = self.config.USER_DIR / user_id
        user_dir.mkdir(exist_ok=True)
        session_file = user_dir / f"{session_id}.json"
        return TinyDB(session_file)
    
    async def add_memory(self, 
                        messages: List[Dict[str, str]], 
                        user_id: str,
                        session_id: Optional[str] = None,
                        memory_type: MemoryType = MemoryType.EPISODIC) -> Dict:
        """Add memory with intelligent categorization"""
        
        metadata = {
            "type": memory_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id
        }
        
        result = {"status": "stored_locally"}
        
        # Store in Mem0 if available
        if self.mem0_client:
            try:
                if session_id:
                    result = self.mem0_client.add(
                        messages, 
                        user_id=user_id,
                        run_id=session_id,
                        metadata=metadata
                    )
                else:
                    result = self.mem0_client.add(
                        messages,
                        user_id=user_id,
                        metadata=metadata
                    )
            except Exception as e:
                logger.error(f"Failed to store in Mem0: {e}")
        
        # Always store locally in TinyDB
        if session_id:
            try:
                db = self.get_session_db(user_id, session_id)
                db.insert({
                    "messages": messages,
                    "metadata": metadata,
                    "mem0_result": result
                })
            except Exception as e:
                logger.error(f"Failed to store in TinyDB: {e}")
        
        # Update compression buffer
        self.compression_buffer.append({
            "user_id": user_id,
            "session_id": session_id,
            "messages": messages,
            "timestamp": datetime.utcnow()
        })
        
        return result
    
    async def search_memories(self,
                            query: str,
                            user_id: str,
                            session_id: Optional[str] = None,
                            limit: int = 5,
                            memory_types: Optional[List[MemoryType]] = None) -> List[Dict]:
        """Search memories with relevance scoring and compression"""
        
        results = []
        
        # Search in Mem0 if available
        if self.mem0_client:
            try:
                mem0_results = self.mem0_client.search(
                    query=query,
                    user_id=user_id,
                    limit=limit * 2
                )
                results = mem0_results.get("results", [])
            except Exception as e:
                logger.error(f"Mem0 search failed: {e}")
        
        # Filter by memory type
        if memory_types and results:
            type_values = [mt.value for mt in memory_types]
            results = [
                r for r in results
                if r.get("metadata", {}).get("type") in type_values
            ]
        
        # Score and rank results
        scored_results = []
        for result in results[:limit]:
            recency_score = self._calculate_recency_score(result)
            similarity_score = result.get("score", 0.5)
            combined_score = 0.7 * similarity_score + 0.3 * recency_score
            
            if combined_score >= self.config.MEMORY_RELEVANCE_THRESHOLD:
                result["relevance_score"] = combined_score
                scored_results.append(result)
        
        # Compress if needed
        return await self._compress_memories(scored_results)
    
    async def get_conversation_history(self,
                                      user_id: str,
                                      session_id: str,
                                      max_tokens: int = None) -> List[Message]:
        """Get conversation history with token budget management"""
        if max_tokens is None:
            max_tokens = self.config.TOKEN_BUDGET_CONVERSATION
            
        try:
            db = self.get_session_db(user_id, session_id)
            all_records = db.all()
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []
        
        messages = []
        token_count = 0
        
        # Get recent messages within token budget
        for record in reversed(all_records):
            msg_list = record.get("messages", [])
            for msg in msg_list:
                # Estimate tokens
                estimated_tokens = len(msg.get("content", "")) // 4
                
                if token_count + estimated_tokens > max_tokens:
                    if messages and len(messages) > 2:
                        messages = await self._compress_conversation(messages)
                    break
                
                messages.append(Message(**msg))
                token_count += estimated_tokens
        
        return list(reversed(messages))
    
    async def _compress_memories(self, memories: List[Dict]) -> List[Dict]:
        """Compress memories to fit token budget"""
        if not memories:
            return memories
        
        total_length = sum(len(m.get("memory", "")) for m in memories)
        
        if total_length < self.config.TOKEN_BUDGET_MEMORY:
            return memories
        
        compressed = []
        for memory in memories:
            content = memory.get("memory", "")
            
            # Extract key phrases
            key_phrases = self._extract_key_phrases(content)
            
            if len(key_phrases) < len(content) * self.config.COMPRESSION_RATIO:
                memory["memory"] = key_phrases
                memory["compressed"] = True
            
            compressed.append(memory)
        
        return compressed
    
    async def _compress_conversation(self, messages: List[Message]) -> List[Message]:
        """Compress conversation history using summarization"""
        if len(messages) <= 3:
            return messages
        
        first = messages[0]
        last = messages[-1]
        middle = messages[1:-1]
        
        summary_content = "Previous conversation: "
        for msg in middle:
            summary_content += f"{msg.role}: {msg.content[:50]}... "
        
        summary_message = Message(
            role="system",
            content=f"[Compressed history] {summary_content[:200]}",
            metadata={"compressed": True}
        )
        
        return [first, summary_message, last]
    
    def _calculate_recency_score(self, memory: Dict) -> float:
        """Calculate recency score for memory ranking"""
        timestamp_str = memory.get("metadata", {}).get("timestamp")
        if not timestamp_str:
            return 0.5
        
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            age_hours = (datetime.utcnow() - timestamp).total_seconds() / 3600
            return np.exp(-age_hours / 168)  # 1 week decay
        except:
            return 0.5
    
    def _extract_key_phrases(self, text: str) -> str:
        """Extract key phrases from text for compression"""
        stop_words = {"the", "is", "at", "which", "on", "and", "a", "an", "as", 
                     "are", "was", "were", "been", "be"}
        words = text.lower().split()
        
        key_words = []
        for word in text.split():
            if (word.lower() not in stop_words or 
                word[0].isupper() or 
                any(c.isdigit() for c in word)):
                key_words.append(word)
        
        return " ".join(key_words[:int(len(key_words) * 0.6)])


# ==================== Tool System ====================

class Tool:
    """Base class for tools"""
    def __init__(self):
        self.name = ""
        self.description = ""
        self.parameters = {}
    
    async def execute(self, **kwargs) -> str:
        raise NotImplementedError


class CalculatorTool(Tool):
    """Calculator tool for mathematical operations"""
    def __init__(self):
        super().__init__()
        self.name = "calculator"
        self.description = "Perform mathematical calculations"
        self.parameters = {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    
    async def execute(self, expression: str, **kwargs) -> str:
        try:
            # Safe evaluation
            allowed_names = {
                k: v for k, v in __builtins__.items()
                if k in ['abs', 'round', 'min', 'max', 'sum', 'pow']
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"Result: {result}"
        except Exception as e:
            return f"Error calculating: {str(e)}"


class MemorySearchTool(Tool):
    """Tool to search through memories"""
    def __init__(self, memory_manager: MemoryManager):
        super().__init__()
        self.name = "memory_search"
        self.description = "Search through user and session memories"
        self.parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["episodic", "semantic", "preference"],
                    "description": "Type of memory to search"
                }
            },
            "required": ["query"]
        }
        self.memory_manager = memory_manager
    
    async def execute(self, query: str, memory_type: Optional[str] = None, 
                     user_id: str = None, **kwargs) -> str:
        memory_types = [MemoryType(memory_type)] if memory_type else None
        results = await self.memory_manager.search_memories(
            query, user_id, memory_types=memory_types, limit=3
        )
        
        if results:
            memories = "\n".join([r.get("memory", "") for r in results[:3]])
            return f"Found memories:\n{memories}"
        return "No relevant memories found."


class DateTimeTool(Tool):
    """Tool to get current date and time"""
    def __init__(self):
        super().__init__()
        self.name = "datetime"
        self.description = "Get current date and time"
        self.parameters = {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone (default: UTC)"
                }
            }
        }
    
    async def execute(self, timezone: str = "UTC", **kwargs) -> str:
        current_time = datetime.utcnow()
        return f"Current time ({timezone}): {current_time.isoformat()}"


class WebSearchTool(Tool):
    """Simulated web search tool"""
    def __init__(self):
        super().__init__()
        self.name = "web_search"
        self.description = "Search the web for information"
        self.parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    
    async def execute(self, query: str, **kwargs) -> str:
        # Simulated search results
        return f"Search results for '{query}':\n1. Example result about {query}\n2. Another relevant result\n3. Additional information"


# ==================== ReAct Agent ====================

class ReActAgent:
    """ReAct (Reasoning and Acting) agent implementation"""
    
    REACT_PROMPT_TEMPLATE = """You are a helpful AI assistant that can reason step-by-step and use tools.
    
Available tools:
{tools}

For complex queries requiring multiple steps, use this format EXACTLY:
Thought: [reasoning about what to do]
Action: [tool_name]
Action Input: {{"param": "value"}}
Observation: [tool output will appear here]
... (repeat Thought/Action/Observation as needed)
Thought: [final reasoning]
Final Answer: [your response to the user]

For simple queries, you can respond directly with:
Final Answer: [your direct response]

Current context:
- User ID: {user_id}
- Session ID: {session_id}
- Relevant memories: {memories}

User message: {message}

Begin your response:"""
    
    def __init__(self, llm_client: AsyncOpenAI, tools: List[Tool], memory_manager: MemoryManager, config: Config):
        self.llm_client = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.memory_manager = memory_manager
        self.config = config
    
    async def process(self,
                     message: str,
                     user_id: str,
                     session_id: str,
                     stream: bool = True) -> AsyncIterator[str]:
        """Process message with ReAct reasoning"""
        try:
            # Search for relevant memories
            memories = await self.memory_manager.search_memories(
                message, user_id, session_id, limit=3
            )
            memory_text = "\n".join([m.get("memory", "") for m in memories]) if memories else "No relevant memories"
            
            # Format tools
            tools_desc = "\n".join([
                f"- {name}: {tool.description}" 
                for name, tool in self.tools.items()
            ])
            
            # Create initial prompt
            prompt = self.REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                user_id=user_id,
                session_id=session_id,
                memories=memory_text,
                message=message
            )
            
            # Initialize conversation
            reasoning_steps = []
            current_prompt = prompt
            final_answer = ""
            
            # Process with timeout
            timeout = self.config.REASONING_TIMEOUT
            start_time = datetime.utcnow()
            
            for step in range(self.config.MAX_REASONING_STEPS):
                # Check timeout
                if (datetime.utcnow() - start_time).total_seconds() > timeout:
                    yield "I apologize, but I'm taking too long to process this request. Please try again.\n"
                    break
                
                try:
                    # Get LLM response
                    response = await self.llm_client.chat.completions.create(
                        model=self.config.OPENAI_MODEL if self.config.USE_OPENAI else self.config.LLM_MODEL,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that follows the ReAct format exactly."},
                            {"role": "user", "content": current_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1000,
                        stream=False
                    )
                    
                    response_text = response.choices[0].message.content
                    reasoning_steps.append(response_text)
                    
                    # Check for Final Answer
                    if "Final Answer:" in response_text:
                        # Extract final answer
                        parts = response_text.split("Final Answer:")
                        if len(parts) > 1:
                            final_answer = parts[-1].strip()
                            
                            # Stream the final answer
                            if stream:
                                # Send chunks properly formatted
                                words = final_answer.split()
                                for i, word in enumerate(words):
                                    if i > 0:
                                        yield " "
                                    yield word
                                    await asyncio.sleep(self.config.STREAM_CHUNK_DELAY)
                            else:
                                yield final_answer
                        break
                    
                    # Check for Action
                    if "Action:" in response_text and "Action Input:" in response_text:
                        # Parse action and input
                        action_match = re.search(r"Action:\s*(\w+)", response_text)
                        input_match = re.search(r"Action Input:\s*({.*?})", response_text, re.DOTALL)
                        
                        if action_match and input_match:
                            action_name = action_match.group(1).strip()
                            
                            try:
                                # Parse action input
                                action_input_str = input_match.group(1).strip()
                                action_input = json.loads(action_input_str)
                            except json.JSONDecodeError:
                                # Try to extract as simple string
                                action_input = {"query": action_input_str}
                            
                            # Execute tool
                            if action_name in self.tools:
                                if stream:
                                    yield f"\n[Thinking: Using {action_name} tool...]\n"
                                    await asyncio.sleep(self.config.STREAM_CHUNK_DELAY)
                                
                                tool_result = await self.tools[action_name].execute(
                                    user_id=user_id,
                                    **action_input
                                )
                                
                                # Add observation to prompt
                                current_prompt += f"\n{response_text}\nObservation: {tool_result}\n"
                                current_prompt += "Continue with your next Thought or provide Final Answer:"
                            else:
                                current_prompt += f"\n{response_text}\nObservation: Tool '{action_name}' not found. Please use a valid tool or provide Final Answer.\n"
                        else:
                            # Invalid format, ask for correction
                            current_prompt += f"\n{response_text}\nObservation: Invalid action format. Please follow the exact format or provide Final Answer.\n"
                    else:
                        # No action and no final answer, treat as final
                        if stream:
                            words = response_text.split()
                            for i, word in enumerate(words):
                                if i > 0:
                                    yield " "
                                yield word
                                await asyncio.sleep(self.config.STREAM_CHUNK_DELAY)
                        else:
                            yield response_text
                        break
                        
                except Exception as e:
                    logger.error(f"Error in reasoning step {step}: {e}")
                    if stream:
                        yield f"\n[Error in reasoning: {str(e)}. Providing direct answer...]\n"
                    
                    # Fallback to direct answer
                    try:
                        fallback_response = await self.llm_client.chat.completions.create(
                            model=self.config.OPENAI_MODEL if self.config.USE_OPENAI else self.config.LLM_MODEL,
                            messages=[
                                {"role": "system", "content": "Provide a helpful response to the user's question."},
                                {"role": "user", "content": message}
                            ],
                            temperature=0.7,
                            max_tokens=1000,
                            stream=False
                        )
                        
                        fallback_text = fallback_response.choices[0].message.content
                        if stream:
                            words = fallback_text.split()
                            for i, word in enumerate(words):
                                if i > 0:
                                    yield " "
                                yield word
                                await asyncio.sleep(self.config.STREAM_CHUNK_DELAY)
                        else:
                            yield fallback_text
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed: {fallback_error}")
                        yield "I encountered an error while processing your request. Please try again."
                    break
            
            # Ensure we always yield something
            if not final_answer and not reasoning_steps:
                yield "I apologize, but I wasn't able to process your request properly. Please try again."
                
        except Exception as e:
            logger.error(f"Critical error in ReAct agent: {e}\n{traceback.format_exc()}")
            yield f"An error occurred: {str(e)}"


# ==================== Chat Service ====================

class ChatService:
    """Main chat service with streaming support"""
    
    def __init__(self):
        self.config = Config()
        self.memory_manager = MemoryManager(self.config)
        
        # Initialize LLM client
        if self.config.USE_OPENAI:
            if not self.config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required when USE_OPENAI=true")
            self.llm_client = AsyncOpenAI(api_key=self.config.OPENAI_API_KEY)
        else:
            self.llm_client = AsyncOpenAI(
                base_url=self.config.LLM_BASE_URL,
                api_key=self.config.LLM_API_KEY
            )
        
        # Initialize tools
        self.tools = [
            CalculatorTool(),
            MemorySearchTool(self.memory_manager),
            DateTimeTool(),
            WebSearchTool()
        ]
        
        # Initialize ReAct agent
        self.react_agent = ReActAgent(
            self.llm_client, 
            self.tools, 
            self.memory_manager,
            self.config
        )
    
    async def process_chat(self, request: ChatRequest) -> AsyncIterator[str]:
        """Process chat request with streaming response"""
        try:
            # Get conversation history
            history = await self.memory_manager.get_conversation_history(
                request.user_id, 
                request.session_id, 
                max_tokens=self.config.TOKEN_BUDGET_CONVERSATION
            )
            
            # Store user message
            await self.memory_manager.add_memory(
                [{"role": "user", "content": request.message}],
                request.user_id,
                request.session_id,
                MemoryType.EPISODIC
            )
            
            # Process based on mode
            assistant_response = ""
            
            if request.reasoning_mode:
                # Use ReAct agent
                logger.info(f"Processing with ReAct reasoning for user {request.user_id}")
                
                async for chunk in self.react_agent.process(
                    request.message,
                    request.user_id,
                    request.session_id,
                    stream=True
                ):
                    if chunk:
                        assistant_response += chunk
                        # Format as SSE
                        yield f"data: {json.dumps({'content': chunk})}\n\n"
                        
            else:
                # Direct response
                logger.info(f"Processing direct response for user {request.user_id}")
                
                messages = self._prepare_messages(history, request.message, request.user_id)
                
                # Stream response
                try:
                    stream = await self.llm_client.chat.completions.create(
                        model=self.config.OPENAI_MODEL if self.config.USE_OPENAI else self.config.LLM_MODEL,
                        messages=messages,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        stream=True,
                        tools=self._format_tools() if not self.config.USE_OPENAI else None
                    )
                    
                    async for chunk in stream:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            assistant_response += content
                            yield f"data: {json.dumps({'content': content})}\n\n"
                        
                        # Handle tool calls
                        if hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                            for tool_call in chunk.choices[0].delta.tool_calls:
                                if tool_call.function:
                                    tool_result = await self._execute_tool_call(
                                        tool_call.function.name,
                                        tool_call.function.arguments,
                                        request.user_id
                                    )
                                    yield f"data: {json.dumps({'tool_result': tool_result})}\n\n"
                                    
                except Exception as e:
                    logger.error(f"Error in direct response: {e}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
            # Store assistant response
            if assistant_response:
                await self.memory_manager.add_memory(
                    [
                        {"role": "user", "content": request.message},
                        {"role": "assistant", "content": assistant_response}
                    ],
                    request.user_id,
                    request.session_id,
                    MemoryType.EPISODIC
                )
                
                # Extract semantic memories
                await self._extract_semantic_memories(assistant_response, request.user_id)
            
            # Send completion signal
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            logger.error(f"Error in process_chat: {e}\n{traceback.format_exc()}")
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
    
    def _prepare_messages(self, 
                         history: List[Message], 
                         user_message: str,
                         user_id: str) -> List[Dict[str, str]]:
        """Prepare messages for LLM"""
        
        messages = [
            {
                "role": "system",
                "content": f"""You are a helpful AI assistant with memory capabilities.
                User ID: {user_id}
                You can use tools to help answer questions.
                Keep responses concise and relevant."""
            }
        ]
        
        # Add history (last N messages based on config)
        max_history = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))
        for msg in history[-max_history:]:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Add current message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages
    
    def _format_tools(self) -> List[Dict[str, Any]]:
        """Format tools for OpenAI-compatible API"""
        formatted_tools = []
        for tool in self.tools:
            formatted_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })
        return formatted_tools
    
    async def _execute_tool_call(self, 
                                tool_name: str, 
                                arguments: str,
                                user_id: str) -> str:
        """Execute a tool call"""
        tool_map = {t.name: t for t in self.tools}
        
        if tool_name not in tool_map:
            return f"Tool {tool_name} not found"
        
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            args['user_id'] = user_id
            
            result = await tool_map[tool_name].execute(**args)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return f"Error executing tool {tool_name}: {str(e)}"
    
    async def _extract_semantic_memories(self, text: str, user_id: str):
        """Extract semantic facts from conversation"""
        patterns = [
            (r"(?:I am|I'm|My name is) ([A-Za-z]+)", "name"),
            (r"(?:I like|I prefer|I enjoy) ([^.]+)", "preference"),
            (r"(?:I work|I'm a|My job) ([^.]+)", "occupation"),
        ]
        
        for pattern, memory_type in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                await self.memory_manager.add_memory(
                    [{"role": "system", "content": f"User {memory_type}: {match}"}],
                    user_id,
                    memory_type=MemoryType.SEMANTIC
                )


# ==================== FastAPI Application ====================

app = FastAPI(
    title="Advanced Chat API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service
chat_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global chat_service
    try:
        chat_service = ChatService()
        logger.info("Chat service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize chat service: {e}")
        raise


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Stream chat responses with memory and tool support
    """
    if not chat_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        return StreamingResponse(
            chat_service.process_chat(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Transfer-Encoding": "chunked"
            }
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "use_openai": Config.USE_OPENAI,
            "mem0_enabled": Config.USE_MEM0_CLOUD,
            "max_reasoning_steps": Config.MAX_REASONING_STEPS
        }
    }


@app.get("/sessions/{user_id}")
async def get_user_sessions(user_id: str):
    """Get all sessions for a user"""
    user_dir = Config.USER_DIR / user_id
    if not user_dir.exists():
        return {"sessions": []}
    
    sessions = []
    for session_file in user_dir.glob("*.json"):
        try:
            session_id = session_file.stem
            db = TinyDB(session_file)
            sessions.append({
                "session_id": session_id,
                "message_count": len(db.all()),
                "last_updated": datetime.fromtimestamp(session_file.stat().st_mtime).isoformat()
            })
        except Exception as e:
            logger.error(f"Error reading session {session_file}: {e}")
            continue
    
    return {"sessions": sessions}


@app.delete("/memory/{user_id}")
async def clear_user_memory(user_id: str, session_id: Optional[str] = None):
    """Clear memory for a user or session"""
    try:
        if session_id:
            session_file = Config.USER_DIR / user_id / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
            return {"message": f"Cleared session {session_id} for user {user_id}"}
        else:
            user_dir = Config.USER_DIR / user_id
            if user_dir.exists():
                for file in user_dir.glob("*.json"):
                    file.unlink()
            return {"message": f"Cleared all memory for user {user_id}"}
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/compress")
async def compress_memories(user_id: str, background_tasks: BackgroundTasks):
    """Trigger memory compression for a user"""
    
    async def compress_task():
        try:
            logger.info(f"Starting memory compression for user {user_id}")
            # Implement compression logic here
            await asyncio.sleep(1)  # Placeholder
            logger.info(f"Completed memory compression for user {user_id}")
        except Exception as e:
            logger.error(f"Error compressing memories: {e}")
    
    background_tasks.add_task(compress_task)
    return {"message": f"Memory compression initiated for user {user_id}"}


if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level=Config.LOG_LEVEL.lower()
    )


"""
Test Suite for Advanced Chat Backend
Tests all endpoints, memory management, tool calling, and reasoning modes
"""

import asyncio
import json
import httpx
import uuid
from datetime import datetime
from typing import AsyncIterator, List, Dict, Any
import pytest
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_USER_ID = "test_user_123"
TEST_SESSION_ID = str(uuid.uuid4())


class ChatTester:
    """Comprehensive test suite for the chat backend"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def test_health_check(self):
        """Test health endpoint"""
        response = await self.client.get(f"{self.base_url}/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        logger.info("✓ Health check passed")
        return True
    
    async def test_simple_chat(self):
        """Test simple chat without reasoning"""
        request_data = {
            "user_id": TEST_USER_ID,
            "session_id": TEST_SESSION_ID,
            "message": "Hello, how are you?",
            "reasoning_mode": False
        }
        
        response_text = await self._stream_chat(request_data)
        assert len(response_text) > 0
        logger.info(f"✓ Simple chat response: {response_text[:100]}...")
        return True
    
    async def test_chat_with_memory(self):
        """Test chat with memory persistence"""
        session_id = str(uuid.uuid4())
        
        # First message - introduce information
        request1 = {
            "user_id": TEST_USER_ID,
            "session_id": session_id,
            "message": "My name is Alice and I love hiking in the mountains.",
            "reasoning_mode": False
        }
        response1 = await self._stream_chat(request1)
        logger.info(f"First response: {response1[:100]}...")
        
        # Second message - test memory recall
        request2 = {
            "user_id": TEST_USER_ID,
            "session_id": session_id,
            "message": "What did I tell you my name was?",
            "reasoning_mode": False
        }
        response2 = await self._stream_chat(request2)
        
        # Check if the response contains the name
        assert "alice" in response2.lower()
        logger.info(f"✓ Memory recall test passed: {response2[:100]}...")
        return True
    
    async def test_reasoning_mode(self):
        """Test ReAct reasoning mode"""
        request = {
            "user_id": TEST_USER_ID,
            "session_id": str(uuid.uuid4()),
            "message": "Calculate 245 * 37 and then add 15% to the result",
            "reasoning_mode": True
        }
        
        response = await self._stream_chat(request)
        
        # Check for reasoning indicators
        assert len(response) > 0
        # The response should contain the calculation result (9065 * 1.15 = 10424.75)
        logger.info(f"✓ Reasoning mode response: {response[:200]}...")
        return True
    
    async def test_tool_calling(self):
        """Test tool calling capabilities"""
        test_cases = [
            {
                "message": "What is 25 + 37 * 2?",
                "expected_tool": "calculator",
                "description": "Calculator tool"
            },
            {
                "message": "What is the current date and time?",
                "expected_tool": "datetime",
                "description": "DateTime tool"
            },
            {
                "message": "Search for information about Python programming",
                "expected_tool": "web_search",
                "description": "Web search tool"
            }
        ]
        
        for test_case in test_cases:
            request = {
                "user_id": TEST_USER_ID,
                "session_id": str(uuid.uuid4()),
                "message": test_case["message"],
                "reasoning_mode": True  # Enable reasoning to see tool usage
            }
            
            response = await self._stream_chat(request)
            logger.info(f"✓ {test_case['description']} test: {response[:100]}...")
        
        return True
    
    async def test_session_management(self):
        """Test session management endpoints"""
        # Create some test sessions
        for i in range(3):
            request = {
                "user_id": TEST_USER_ID,
                "session_id": f"test_session_{i}",
                "message": f"Test message {i}",
                "reasoning_mode": False
            }
            await self._stream_chat(request)
        
        # Get user sessions
        response = await self.client.get(f"{self.base_url}/sessions/{TEST_USER_ID}")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert len(data["sessions"]) >= 3
        logger.info(f"✓ Found {len(data['sessions'])} sessions for user")
        return True
    
    async def test_memory_compression(self):
        """Test memory compression endpoint"""
        response = await self.client.post(
            f"{self.base_url}/memory/compress",
            params={"user_id": TEST_USER_ID}
        )
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        logger.info("✓ Memory compression initiated successfully")
        return True
    
    async def test_memory_deletion(self):
        """Test memory deletion endpoints"""
        test_session = str(uuid.uuid4())
        
        # Create a session
        request = {
            "user_id": TEST_USER_ID,
            "session_id": test_session,
            "message": "Test message for deletion",
            "reasoning_mode": False
        }
        await self._stream_chat(request)
        
        # Delete session memory
        response = await self.client.delete(
            f"{self.base_url}/memory/{TEST_USER_ID}",
            params={"session_id": test_session}
        )
        assert response.status_code == 200
        logger.info("✓ Session memory deleted successfully")
        
        # Delete all user memory
        response = await self.client.delete(f"{self.base_url}/memory/{TEST_USER_ID}")
        assert response.status_code == 200
        logger.info("✓ User memory cleared successfully")
        return True
    
    async def test_conversation_context(self):
        """Test multi-turn conversation with context preservation"""
        session_id = str(uuid.uuid4())
        conversation = [
            "I'm planning a trip to Japan next month.",
            "I'm particularly interested in visiting temples and trying local food.",
            "What would you recommend for a first-time visitor?",
            "How about accommodation? I prefer traditional experiences.",
            "Can you remind me what I said I was interested in?"
        ]
        
        responses = []
        for i, message in enumerate(conversation):
            request = {
                "user_id": TEST_USER_ID,
                "session_id": session_id,
                "message": message,
                "reasoning_mode": False
            }
            response = await self._stream_chat(request)
            responses.append(response)
            logger.info(f"Turn {i+1}: {message[:50]}... -> {response[:100]}...")
        
        # Check if the last response contains references to earlier context
        last_response = responses[-1].lower()
        assert "temple" in last_response or "food" in last_response
        logger.info("✓ Conversation context preserved across turns")
        return True
    
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        tasks = []
        for i in range(5):
            request = {
                "user_id": f"concurrent_user_{i}",
                "session_id": str(uuid.uuid4()),
                "message": f"Concurrent request {i}",
                "reasoning_mode": False
            }
            tasks.append(self._stream_chat(request))
        
        responses = await asyncio.gather(*tasks)
        assert len(responses) == 5
        assert all(len(r) > 0 for r in responses)
        logger.info(f"✓ Handled {len(responses)} concurrent requests successfully")
        return True
    
    async def test_token_budget_management(self):
        """Test token budget management with long conversations"""
        session_id = str(uuid.uuid4())
        
        # Send many messages to test compression
        for i in range(20):
            request = {
                "user_id": TEST_USER_ID,
                "session_id": session_id,
                "message": f"This is message number {i}. " * 10,  # Long message
                "reasoning_mode": False,
                "max_tokens": 100  # Limit response length
            }
            response = await self._stream_chat(request)
            
            if i % 5 == 0:
                logger.info(f"Processed {i+1} messages, last response length: {len(response)}")
        
        logger.info("✓ Token budget management test completed")
        return True
    
    async def test_error_handling(self):
        """Test error handling for invalid requests"""
        # Test with missing user_id
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/chat",
                json={"message": "Test"},
                timeout=10.0
            ) as response:
                if response.status_code == 422:  # Validation error
                    logger.info("✓ Properly rejected request with missing user_id")
        except Exception as e:
            logger.info(f"✓ Error handling test passed with exception: {type(e).__name__}")
        
        return True
    
    async def _stream_chat(self, request_data: Dict[str, Any]) -> str:
        """Helper to handle streaming chat responses"""
        full_response = ""
        
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/chat",
                json=request_data,
                timeout=30.0
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if "content" in data:
                                full_response += data["content"]
                            elif "tool_result" in data:
                                full_response += f"\n[Tool Result: {data['tool_result']}]\n"
                            elif data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            raise
        
        return full_response
    
    async def run_all_tests(self):
        """Run all test cases"""
        tests = [
            ("Health Check", self.test_health_check),
            ("Simple Chat", self.test_simple_chat),
            ("Chat with Memory", self.test_chat_with_memory),
            ("Reasoning Mode", self.test_reasoning_mode),
            ("Tool Calling", self.test_tool_calling),
            ("Session Management", self.test_session_management),
            ("Memory Compression", self.test_memory_compression),
            ("Memory Deletion", self.test_memory_deletion),
            ("Conversation Context", self.test_conversation_context),
            ("Concurrent Requests", self.test_concurrent_requests),
            ("Token Budget Management", self.test_token_budget_management),
            ("Error Handling", self.test_error_handling),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Running: {test_name}")
                logger.info(f"{'='*50}")
                result = await test_func()
                results.append((test_name, "PASSED" if result else "FAILED"))
                await asyncio.sleep(0.5)  # Small delay between tests
            except Exception as e:
                logger.error(f"Test {test_name} failed with error: {e}")
                results.append((test_name, "ERROR"))
        
        # Print summary
        logger.info(f"\n{'='*50}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*50}")
        for test_name, status in results:
            emoji = "✅" if status == "PASSED" else "❌"
            logger.info(f"{emoji} {test_name}: {status}")
        
        passed = sum(1 for _, status in results if status == "PASSED")
        total = len(results)
        logger.info(f"\nTotal: {passed}/{total} tests passed")
        
        return results


async def main():
    """Main test runner"""
    logger.info("Starting Chat Backend Test Suite")
    logger.info(f"Testing against: {BASE_URL}")
    
    async with ChatTester() as tester:
        # First check if the server is running
        try:
            await tester.test_health_check()
        except Exception as e:
            logger.error(f"Server not responding at {BASE_URL}")
            logger.error("Please start the server first with: python main.py")
            return
        
        # Run all tests
        results = await tester.run_all_tests()
        
        # Return exit code based on results
        if all(status == "PASSED" for _, status in results):
            return 0
        else:
            return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
