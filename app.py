"""
AI Backend with Normal and Agentic Workflows
FastAPI backend supporting both single AI responses and multi-agent workflows
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Iterator
from datetime import datetime
import asyncio
import json
import uuid
import os
import time
from contextlib import asynccontextmanager

from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.workflow import Workflow, RunEvent
from agno.storage.sqlite import SqliteStorage
from agno.utils.log import logger

# Try to import memory v2, fallback to simpler approach if not available
try:
    from agno.memory.v2.db.sqlite import SqliteMemoryDb
    from agno.memory.v2.memory import Memory
    MEMORY_V2_AVAILABLE = True
except ImportError:
    MEMORY_V2_AVAILABLE = False
    logger.warning("Memory v2 not available, using simplified storage-only approach")

# Data Models
class NormalAIRequest(BaseModel):
    persona: str = Field(..., description="The persona/role for the AI agent")
    task: str = Field(..., description="The task or question to be processed")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")

class AgentConfig(BaseModel):
    name: str = Field(..., description="Agent name")
    persona: str = Field(..., description="Agent persona/role")
    task: str = Field(..., description="Agent's specific task")

class AgenticAIRequest(BaseModel):
    agents: List[AgentConfig] = Field(..., description="List of 3-4 agents configuration")
    initial_task: str = Field(..., description="Initial task for the workflow")
    session_id: Optional[str] = Field(None, description="Session ID for workflow continuity")

class FeedbackRequest(BaseModel):
    session_id: str = Field(..., description="Session ID")
    agent_index: int = Field(..., description="Index of the agent (0-based)")
    feedback: str = Field(..., description="User feedback on agent response")

class AIResponse(BaseModel):
    response: str
    agent_name: Optional[str] = None
    session_id: str
    timestamp: str

class AgenticWorkflowResponse(BaseModel):
    current_agent_index: int
    current_agent_name: str
    response: str
    session_id: str
    is_complete: bool
    timestamp: str
    feedback_count: int
    max_feedback: int = 3

class WorkflowState(BaseModel):
    session_id: str
    current_agent_index: int
    agents: List[AgentConfig]
    agent_responses: Dict[int, str] = {}
    feedback_history: Dict[int, List[str]] = {}
    common_feedback_memory: List[str] = []
    initial_task: str = ""
    is_complete: bool = False
    created_at: str

# Global storage for workflow states
workflow_states: Dict[str, WorkflowState] = {}

class MultiAgentWorkflow(Workflow):
    """Custom workflow for sequential agent execution with feedback"""
    
    def __init__(self, agents_config: List[AgentConfig], session_id: str, initial_task: str):
        super().__init__(session_id=session_id)
        self.agents_config = agents_config
        self.initial_task = initial_task
        self.agents: List[Agent] = []
        self.setup_agents()
        
    def setup_agents(self):
        """Initialize all agents with DeepSeek model and storage"""
        for i, config in enumerate(self.agents_config):
            # Create storage for session history - use unique table per agent and session
            agent_storage = SqliteStorage(
                table_name=f"agent_{i}_{self.session_id.replace('-', '_')}",
                db_file="tmp/agent_sessions.db"
            )
            
            if MEMORY_V2_AVAILABLE:
                # Use Memory v2 if available
                try:
                    memory_db = SqliteMemoryDb(
                        table_name=f"agent_{i}_memory_{self.session_id.replace('-', '_')}",
                        db_file="tmp/agent_memory.db"
                    )
                    # Create memory with OpenAIChat model for memory operations
                    agent_memory = Memory(
                        model=OpenAIChat(id="gpt-3.5-turbo-16k"),
                        db=memory_db
                    )
                    
                    agent = Agent(
                        name=config.name,
                        model=OpenAIChat(id="gpt-3.5-turbo-16k"),
                        description=config.persona,
                        instructions=[
                            config.task,
                            "Provide detailed, thoughtful responses.",
                            "Consider any previous agent responses and feedback when available.",
                            "Be collaborative and build upon previous work.",
                            "Always complete your analysis fully before stopping.",
                            "Focus on the specific task assigned to you."
                        ],
                        memory=agent_memory,
                        storage=agent_storage,
                        session_id=f"{self.session_id}_agent_{i}",
                        markdown=True,
                        add_history_to_messages=True,
                        num_history_responses=5,
                        enable_user_memories=True
                    )
                except Exception as e:
                    logger.warning(f"Failed to create memory for agent {i}, falling back to storage-only: {e}")
                    # Fallback to storage-only approach
                    agent = self._create_storage_only_agent(config, agent_storage, i)
            else:
                # Use storage-only approach
                agent = self._create_storage_only_agent(config, agent_storage, i)
            
            self.agents.append(agent)
    
    def _create_storage_only_agent(self, config: AgentConfig, storage: SqliteStorage, index: int) -> Agent:
        """Create agent with storage only (no complex memory)"""
        return Agent(
            name=config.name,
            model=OpenAIChat(id="gpt-3.5-turbo-16k"),
            description=config.persona,
            instructions=[
                config.task,
                "Provide detailed, thoughtful responses.",
                "Consider any previous agent responses and feedback when available.",
                "Be collaborative and build upon previous work.",
                "Remember the context from previous interactions in this session.",
                "Always complete your analysis fully before stopping.",
                "Focus on the specific task assigned to you."
            ],
            storage=storage,
            session_id=f"{self.session_id}_agent_{index}",
            markdown=True,
            add_history_to_messages=True,
            num_history_responses=5,
            # Use built-in memory features without external memory system
            read_chat_history=True
        )
    
    def run_agent_with_context(self, agent_index: int, context: str, max_retries: int = 3) -> RunResponse:
        """Run a specific agent with context and ensure completion"""
        agent = self.agents[agent_index]
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Running agent {agent_index} ({agent.name}) - attempt {attempt + 1}")
                
                # Run the agent and wait for completion
                run_response: RunResponse = agent.run(context)
                
                # Validate the response
                if run_response and run_response.content and len(run_response.content.strip()) > 50:
                    logger.info(f"Agent {agent_index} completed successfully with {len(run_response.content)} characters")
                    return run_response
                else:
                    logger.warning(f"Agent {agent_index} returned insufficient content: {len(run_response.content if run_response.content else '')} characters")
                    
            except Exception as e:
                logger.error(f"Agent {agent_index} failed on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)  # Brief pause before retry
        
        raise Exception(f"Agent {agent_index} failed after {max_retries} attempts")

# FastAPI App Setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting AI Backend with Normal and Agentic workflows")
    # Ensure directories exist
    os.makedirs("tmp", exist_ok=True)
    yield
    # Shutdown
    logger.info("Shutting down AI Backend")

app = FastAPI(
    title="AI Backend with Normal and Agentic Workflows",
    description="FastAPI backend supporting both single AI responses and multi-agent workflows using DeepSeek",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize storage for normal AI
normal_ai_storage = SqliteStorage(
    table_name="normal_ai_sessions", 
    db_file="tmp/ai_backend.db"
)

def get_session_id() -> str:
    """Generate a new session ID if not provided"""
    return str(uuid.uuid4())

def create_normal_agent(persona: str, session_id: str) -> Agent:
    """Create a normal AI agent with the given persona"""
    if MEMORY_V2_AVAILABLE:
        try:
            # Try to use Memory v2
            memory_db = SqliteMemoryDb(
                table_name=f"normal_ai_memory_{session_id.replace('-', '_')}",
                db_file="tmp/normal_ai_memory.db"
            )
            memory = Memory(
                model=OpenAIChat(id="gpt-3.5-turbo-16k"),
                db=memory_db
            )
            
            return Agent(
                model=OpenAIChat(id="gpt-3.5-turbo-16k"),
                description=persona,
                instructions=[
                    "Provide helpful, accurate, and engaging responses.",
                    "Maintain the persona and role specified.",
                    "Be conversational and friendly."
                ],
                session_id=session_id,
                storage=normal_ai_storage,
                memory=memory,
                markdown=True,
                add_history_to_messages=True,
                num_history_responses=3,
                enable_user_memories=True
            )
        except Exception as e:
            logger.warning(f"Failed to create memory for normal agent, using storage-only: {e}")
    
    # Fallback to storage-only approach
    return Agent(
        model=OpenAIChat(id="gpt-3.5-turbo-16k"),
        description=persona,
        instructions=[
            "Provide helpful, accurate, and engaging responses.",
            "Maintain the persona and role specified.",
            "Be conversational and friendly.",
            "Remember our conversation history to provide context-aware responses."
        ],
        session_id=session_id,
        storage=normal_ai_storage,
        markdown=True,
        add_history_to_messages=True,
        num_history_responses=3,
        read_chat_history=True
    )

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AI Backend with Normal and Agentic Workflows is running!",
        "memory_v2_available": MEMORY_V2_AVAILABLE,
        "active_workflows": len(workflow_states),
        "endpoints": {
            "normal_ai": "/api/v1/normal-ai",
            "agentic_ai": "/api/v1/agentic-ai",
            "agentic_feedback": "/api/v1/agentic-ai/feedback",
            "agentic_next": "/api/v1/agentic-ai/next",
            "workflow_status": "/api/v1/agentic-ai/status/{session_id}"
        }
    }

@app.post("/api/v1/normal-ai", response_model=AIResponse)
async def normal_ai(request: NormalAIRequest):
    """
    Normal AI endpoint: takes a persona and task, returns a response
    """
    try:
        session_id = request.session_id or get_session_id()
        
        # Create agent with the specified persona
        agent = create_normal_agent(request.persona, session_id)
        
        # Get response from agent
        response: RunResponse = agent.run(request.task)
        
        return AIResponse(
            response=response.content,
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in normal AI endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/v1/agentic-ai", response_model=AgenticWorkflowResponse)
async def start_agentic_workflow(request: AgenticAIRequest):
    """
    Start an agentic AI workflow with sequential agents
    """
    try:
        session_id = request.session_id or get_session_id()
        
        # Validate agents configuration
        if len(request.agents) < 3 or len(request.agents) > 4:
            raise HTTPException(
                status_code=400, 
                detail="Agentic workflow requires 3-4 agents"
            )
        
        # Initialize workflow state
        workflow_state = WorkflowState(
            session_id=session_id,
            current_agent_index=0,
            agents=request.agents,
            initial_task=request.initial_task,
            created_at=datetime.now().isoformat()
        )
        
        # Initialize feedback history for all agents
        for i in range(len(request.agents)):
            workflow_state.feedback_history[i] = []
        
        # Store workflow state
        workflow_states[session_id] = workflow_state
        
        # Create workflow and run first agent
        workflow = MultiAgentWorkflow(request.agents, session_id, request.initial_task)
        
        # Prepare comprehensive context for first agent
        context = f"""
WORKFLOW SESSION: {session_id}
AGENT ROLE: {request.agents[0].name} (First Agent in Sequence)

INITIAL TASK:
{request.initial_task}

YOUR SPECIFIC RESPONSIBILITY:
{request.agents[0].task}

INSTRUCTIONS:
- You are the first agent in a {len(request.agents)}-agent collaborative workflow
- Provide a thorough analysis/response to the initial task
- Your response will be passed to the next agent: {request.agents[1].name}
- Be comprehensive and detailed as subsequent agents will build upon your work
- Complete your analysis fully before finishing
"""
        
        # Run first agent with proper completion checking
        logger.info(f"Starting workflow {session_id} with agent 0: {request.agents[0].name}")
        response = workflow.run_agent_with_context(0, context)
        
        # Store response
        workflow_state.agent_responses[0] = response.content
        
        logger.info(f"Agent 0 completed with response length: {len(response.content)}")
        
        return AgenticWorkflowResponse(
            current_agent_index=0,
            current_agent_name=request.agents[0].name,
            response=response.content,
            session_id=session_id,
            is_complete=False,
            timestamp=datetime.now().isoformat(),
            feedback_count=0
        )
        
    except Exception as e:
        logger.error(f"Error in agentic AI workflow start: {str(e)}")
        # Clean up failed workflow state
        if session_id in workflow_states:
            del workflow_states[session_id]
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/v1/agentic-ai/feedback", response_model=AgenticWorkflowResponse)
async def provide_feedback(request: FeedbackRequest):
    """
    Provide feedback to current agent and get refined response
    """
    try:
        if request.session_id not in workflow_states:
            raise HTTPException(status_code=404, detail="Workflow session not found")
        
        workflow_state = workflow_states[request.session_id]
        
        if workflow_state.is_complete:
            raise HTTPException(status_code=400, detail="Workflow is already complete")
        
        if request.agent_index != workflow_state.current_agent_index:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid agent index. Current agent is {workflow_state.current_agent_index}"
            )
        
        # Check feedback limit
        feedback_count = len(workflow_state.feedback_history[request.agent_index])
        if feedback_count >= 3:
            raise HTTPException(
                status_code=400, 
                detail="Maximum feedback limit (3) reached for this agent"
            )
        
        # Add feedback to history and common memory
        workflow_state.feedback_history[request.agent_index].append(request.feedback)
        workflow_state.common_feedback_memory.append(
            f"Agent {request.agent_index} ({workflow_state.agents[request.agent_index].name}) feedback: {request.feedback}"
        )
        
        # Recreate workflow and get refined response
        workflow = MultiAgentWorkflow(workflow_state.agents, request.session_id, workflow_state.initial_task)
        
        # Prepare comprehensive context with feedback
        context_parts = [f"WORKFLOW SESSION: {request.session_id}"]
        context_parts.append(f"AGENT ROLE: {workflow_state.agents[request.agent_index].name}")
        
        if request.agent_index == 0:
            context_parts.append("INITIAL TASK:")
            context_parts.append(workflow_state.initial_task)
        else:
            # Include previous agent responses
            context_parts.append("PREVIOUS AGENT RESPONSES:")
            for i in range(request.agent_index):
                if i in workflow_state.agent_responses:
                    context_parts.append(f"\nAgent {i} ({workflow_state.agents[i].name}):")
                    context_parts.append(workflow_state.agent_responses[i])
        
        # Add feedback context
        context_parts.append(f"\nUSER FEEDBACK ON YOUR PREVIOUS RESPONSE:")
        context_parts.append(request.feedback)
        context_parts.append("\nPLEASE REFINE YOUR RESPONSE BASED ON THIS FEEDBACK:")
        context_parts.append("- Address the specific points mentioned in the feedback")
        context_parts.append("- Improve or expand your previous response")
        context_parts.append("- Maintain consistency with the overall workflow goal")
        
        if workflow_state.common_feedback_memory:
            context_parts.append("\nCOMMON FEEDBACK MEMORY:")
            context_parts.extend(workflow_state.common_feedback_memory[:-1])  # Exclude the current feedback
        
        context = "\n".join(context_parts)
        
        logger.info(f"Providing feedback to agent {request.agent_index}: {request.feedback[:100]}...")
        response = workflow.run_agent_with_context(request.agent_index, context)
        
        # Update stored response
        workflow_state.agent_responses[request.agent_index] = response.content
        
        logger.info(f"Agent {request.agent_index} refined response length: {len(response.content)}")
        
        return AgenticWorkflowResponse(
            current_agent_index=request.agent_index,
            current_agent_name=workflow_state.agents[request.agent_index].name,
            response=response.content,
            session_id=request.session_id,
            is_complete=False,
            timestamp=datetime.now().isoformat(),
            feedback_count=len(workflow_state.feedback_history[request.agent_index])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in feedback endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/v1/agentic-ai/next", response_model=AgenticWorkflowResponse)
async def next_agent(session_id: str):
    """
    Move to the next agent in the workflow
    """
    try:
        if session_id not in workflow_states:
            raise HTTPException(status_code=404, detail="Workflow session not found")
        
        workflow_state = workflow_states[session_id]
        
        if workflow_state.is_complete:
            raise HTTPException(status_code=400, detail="Workflow is already complete")
        
        # Ensure current agent has completed
        current_index = workflow_state.current_agent_index
        if current_index not in workflow_state.agent_responses:
            raise HTTPException(
                status_code=400, 
                detail=f"Current agent {current_index} has not completed yet"
            )
        
        # Move to next agent
        next_agent_index = workflow_state.current_agent_index + 1
        
        if next_agent_index >= len(workflow_state.agents):
            # Mark workflow as complete
            workflow_state.is_complete = True
            
            logger.info(f"Workflow {session_id} completed with {len(workflow_state.agents)} agents")
            
            return AgenticWorkflowResponse(
                current_agent_index=workflow_state.current_agent_index,
                current_agent_name=workflow_state.agents[workflow_state.current_agent_index].name,
                response=workflow_state.agent_responses.get(workflow_state.current_agent_index, ""),
                session_id=session_id,
                is_complete=True,
                timestamp=datetime.now().isoformat(),
                feedback_count=len(workflow_state.feedback_history.get(workflow_state.current_agent_index, []))
            )
        
        # Update current agent index
        workflow_state.current_agent_index = next_agent_index
        
        # Create workflow and run next agent
        workflow = MultiAgentWorkflow(workflow_state.agents, session_id, workflow_state.initial_task)
        
        # Prepare comprehensive context for next agent
        context_parts = [f"WORKFLOW SESSION: {session_id}"]
        context_parts.append(f"AGENT ROLE: {workflow_state.agents[next_agent_index].name} (Agent {next_agent_index + 1} of {len(workflow_state.agents)})")
        
        context_parts.append("ORIGINAL TASK:")
        context_parts.append(workflow_state.initial_task)
        
        context_parts.append("\nYOUR SPECIFIC RESPONSIBILITY:")
        context_parts.append(workflow_state.agents[next_agent_index].task)
        
        # Include all previous agent responses
        context_parts.append("\nPREVIOUS AGENT WORK (build upon this):")
        for i in range(next_agent_index):
            if i in workflow_state.agent_responses:
                context_parts.append(f"\n=== Agent {i} ({workflow_state.agents[i].name}) ===")
                context_parts.append(workflow_state.agent_responses[i])
        
        # Add common feedback memory
        if workflow_state.common_feedback_memory:
            context_parts.append("\nCOMMON FEEDBACK MEMORY:")
            context_parts.extend(workflow_state.common_feedback_memory)
        
        context_parts.append(f"\nINSTRUCTIONS:")
        context_parts.append("- Build upon the previous agents' work")
        context_parts.append("- Address your specific responsibility thoroughly") 
        context_parts.append("- Maintain consistency with the overall workflow goal")
        context_parts.append("- Provide comprehensive analysis before finishing")
        
        if next_agent_index < len(workflow_state.agents) - 1:
            next_agent_name = workflow_state.agents[next_agent_index + 1].name
            context_parts.append(f"- Your work will be passed to: {next_agent_name}")
        else:
            context_parts.append("- You are the final agent, provide conclusive recommendations")
        
        context = "\n".join(context_parts)
        
        logger.info(f"Moving to agent {next_agent_index}: {workflow_state.agents[next_agent_index].name}")
        response = workflow.run_agent_with_context(next_agent_index, context)
        
        # Store response
        workflow_state.agent_responses[next_agent_index] = response.content
        
        logger.info(f"Agent {next_agent_index} completed with response length: {len(response.content)}")
        
        return AgenticWorkflowResponse(
            current_agent_index=next_agent_index,
            current_agent_name=workflow_state.agents[next_agent_index].name,
            response=response.content,
            session_id=session_id,
            is_complete=False,
            timestamp=datetime.now().isoformat(),
            feedback_count=0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in next agent endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/v1/agentic-ai/status/{session_id}")
async def get_workflow_status(session_id: str):
    """
    Get the current status of an agentic workflow
    """
    if session_id not in workflow_states:
        raise HTTPException(status_code=404, detail="Workflow session not found")
    
    workflow_state = workflow_states[session_id]
    
    return {
        "session_id": session_id,
        "current_agent_index": workflow_state.current_agent_index,
        "current_agent_name": workflow_state.agents[workflow_state.current_agent_index].name if not workflow_state.is_complete else None,
        "total_agents": len(workflow_state.agents),
        "is_complete": workflow_state.is_complete,
        "agent_responses": workflow_state.agent_responses,
        "feedback_history": workflow_state.feedback_history,
        "common_feedback_memory": workflow_state.common_feedback_memory,
        "initial_task": workflow_state.initial_task,
        "created_at": workflow_state.created_at,
        "completion_status": {
            f"agent_{i}": i in workflow_state.agent_responses 
            for i in range(len(workflow_state.agents))
        }
    }

@app.delete("/api/v1/agentic-ai/{session_id}")
async def delete_workflow(session_id: str):
    """
    Delete a workflow session
    """
    if session_id not in workflow_states:
        raise HTTPException(status_code=404, detail="Workflow session not found")
    
    del workflow_states[session_id]
    return {"message": f"Workflow session {session_id} deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    
    # Set environment variable if not already set (for testing)
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Warning: DEEPSEEK_API_KEY environment variable not set")
        print("Please set it before running: export DEEPSEEK_API_KEY='your-api-key'")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )