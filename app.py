#!/usr/bin/env python3
"""
Updated AI Backend with Normal and Agentic Workflows
FastAPI backend supporting both single AI responses and multi-agent workflows
Now includes pre-agent instructions support
"""

import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY environment variable is required")

# Pydantic Models
class Agent(BaseModel):
    name: str
    persona: str
    task: str

class AgenticRequest(BaseModel):
    agents: List[Agent]
    initial_task: str

class NormalAIRequest(BaseModel):
    persona: str
    task: str
    max_tokens: Optional[int] = 4000
    temperature: Optional[float] = 0.7

class FeedbackRequest(BaseModel):
    session_id: str
    agent_index: int
    feedback: str

class NextAgentRequest(BaseModel):
    session_id: str
    additional_context: Optional[str] = None  # NEW: Support for pre-agent instructions
    force_next: Optional[bool] = False

# Global storage for sessions
active_sessions: Dict[str, Dict] = {}
memory_v2_data: Dict[str, Any] = {}

app = FastAPI(
    title="AI Backend with Pre-Agent Instructions",
    description="FastAPI backend supporting both single AI responses and multi-agent workflows with pre-agent instructions",
    version="2.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def call_deepseek_api(messages: List[Dict], max_tokens: int = 4000, temperature: float = 0.7) -> str:
    """Call DeepSeek API with messages"""
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise HTTPException(status_code=500, detail=f"DeepSeek API error: {response.status_code}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API call failed: {str(e)}")

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Backend with Pre-Agent Instructions",
        "version": "2.1.0",
        "memory_v2_available": True,
        "active_workflows": len(active_sessions),
        "features": [
            "normal_ai",
            "agentic_ai", 
            "feedback_mechanism",
            "pre_agent_instructions",  # NEW FEATURE
            "session_management"
        ]
    }

@app.post("/api/v1/normal-ai")
async def normal_ai(request: NormalAIRequest):
    """Single AI response endpoint"""
    try:
        messages = [
            {"role": "system", "content": request.persona},
            {"role": "user", "content": request.task}
        ]
        
        response_content = call_deepseek_api(
            messages, 
            request.max_tokens, 
            request.temperature
        )
        
        return {
            "response": response_content,
            "session_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "model": "deepseek-chat",
            "tokens_used": len(response_content.split())  # Approximate
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/agentic-ai")
async def start_agentic_workflow(request: AgenticRequest, background_tasks: BackgroundTasks):
    """Start a new agentic workflow"""
    try:
        session_id = str(uuid.uuid4())
        
        # Initialize session
        session_data = {
            "session_id": session_id,
            "agents": [agent.dict() for agent in request.agents],
            "initial_task": request.initial_task,
            "current_agent_index": 0,
            "agent_responses": {},
            "conversation_history": [],
            "completion_status": {},
            "feedback_history": {},
            "pre_agent_instructions": {},  # NEW: Store pre-agent instructions
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        active_sessions[session_id] = session_data
        
        # Execute first agent
        background_tasks.add_task(execute_agent, session_id, 0, request.initial_task)
        
        return {
            "session_id": session_id,
            "status": "started",
            "current_agent_name": request.agents[0].name,
            "total_agents": len(request.agents),
            "message": "Workflow started successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def execute_agent(session_id: str, agent_index: int, task: str, additional_context: str = None):
    """Execute a specific agent with optional pre-agent instructions"""
    try:
        session = active_sessions.get(session_id)
        if not session:
            return
            
        agent = session["agents"][agent_index]
        
        # Build the complete task context
        full_task = task
        
        # Add conversation history context
        if session["conversation_history"]:
            context_summary = "\n\nPREVIOUS CONTEXT:\n"
            for i, prev_response in enumerate(session["conversation_history"][-3:]):  # Last 3 responses
                context_summary += f"Agent {prev_response['agent_index']} ({prev_response['agent_name']}): {prev_response['response'][:300]}...\n"
            full_task += context_summary
        
        # Add pre-agent instructions if provided
        if additional_context:
            full_task += f"\n\nADDITIONAL INSTRUCTIONS: {additional_context}"
            # Store the pre-agent instructions
            session["pre_agent_instructions"][agent_index] = additional_context
        
        # Add feedback history context if exists
        if str(agent_index) in session.get("feedback_history", {}):
            feedback_context = "\n\nPREVIOUS FEEDBACK TO CONSIDER:\n"
            for feedback in session["feedback_history"][str(agent_index)]:
                feedback_context += f"- {feedback}\n"
            full_task += feedback_context
        
        messages = [
            {"role": "system", "content": agent["persona"]},
            {"role": "user", "content": full_task}
        ]
        
        # Call AI API
        response_content = call_deepseek_api(messages)
        
        # Store response
        session["agent_responses"][str(agent_index)] = response_content
        session["completion_status"][f"agent_{agent_index}"] = True
        
        # Update conversation history
        session["conversation_history"].append({
            "agent_index": agent_index,
            "agent_name": agent["name"],
            "task": full_task,
            "response": response_content,
            "timestamp": datetime.now().isoformat(),
            "pre_agent_instructions": additional_context  # NEW: Track pre-agent instructions
        })
        
        # Update session
        active_sessions[session_id] = session
        
    except Exception as e:
        print(f"Error executing agent {agent_index}: {e}")
        if session_id in active_sessions:
            active_sessions[session_id]["status"] = "error"
            active_sessions[session_id]["error"] = str(e)

@app.post("/api/v1/agentic-ai/next")
async def next_agent(request: NextAgentRequest, background_tasks: BackgroundTasks):
    """Move to the next agent with optional pre-agent instructions"""
    try:
        session = active_sessions.get(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        current_index = session["current_agent_index"]
        next_index = current_index + 1
        
        if next_index >= len(session["agents"]):
            return {
                "status": "workflow_complete", 
                "message": "All agents have completed",
                "session_id": request.session_id
            }
        
        # Check if current agent is complete (unless force_next is True)
        if not request.force_next:
            current_agent_key = f"agent_{current_index}"
            if not session["completion_status"].get(current_agent_key, False):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Current agent {current_index} has not completed yet"
                )
        
        # Update session for next agent
        session["current_agent_index"] = next_index
        active_sessions[request.session_id] = session
        
        # Build task for next agent
        next_agent = session["agents"][next_index]
        
        # Use the accumulated context from previous agents
        task_context = session["initial_task"]
        
        if session["conversation_history"]:
            task_context += "\n\nBUILDING ON PREVIOUS WORK:\n"
            for entry in session["conversation_history"]:
                task_context += f"\n{entry['agent_name']}: {entry['response'][:400]}...\n"
        
        # Execute next agent with optional pre-agent instructions
        background_tasks.add_task(
            execute_agent, 
            request.session_id, 
            next_index, 
            task_context,
            request.additional_context  # NEW: Pass pre-agent instructions
        )
        
        return {
            "status": "moved_to_next",
            "current_agent_index": next_index,
            "current_agent_name": next_agent["name"],
            "session_id": request.session_id,
            "pre_agent_instructions": request.additional_context,  # NEW: Confirm instructions received
            "message": f"Moved to agent {next_index}: {next_agent['name']}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/agentic-ai/feedback")
async def provide_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """Provide feedback to improve an agent's response"""
    try:
        session = active_sessions.get(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        agent_index = request.agent_index
        
        # Validate agent index
        if agent_index >= len(session["agents"]) or agent_index < 0:
            raise HTTPException(status_code=400, detail="Invalid agent index")
        
        # Store feedback in history
        feedback_key = str(agent_index)
        if feedback_key not in session["feedback_history"]:
            session["feedback_history"][feedback_key] = []
        session["feedback_history"][feedback_key].append(request.feedback)
        
        # Get the original task for this agent
        agent = session["agents"][agent_index]
        original_task = session["initial_task"]
        
        # Build context including feedback
        if session["conversation_history"]:
            for entry in session["conversation_history"]:
                if entry["agent_index"] == agent_index:
                    original_task = entry["task"]
                    break
        
        # Add feedback to the task
        improved_task = f"{original_task}\n\nUSER FEEDBACK TO INCORPORATE: {request.feedback}\n\nPlease revise your previous response based on this feedback while maintaining quality and accuracy."
        
        # Get any pre-agent instructions for this agent
        pre_instructions = session.get("pre_agent_instructions", {}).get(agent_index)
        
        # Re-execute agent with feedback
        background_tasks.add_task(
            execute_agent, 
            request.session_id, 
            agent_index, 
            improved_task,
            pre_instructions  # Maintain pre-agent instructions if they existed
        )
        
        # Update session
        active_sessions[request.session_id] = session
        
        return {
            "status": "feedback_received",
            "agent_index": agent_index,
            "agent_name": agent["name"],
            "feedback": request.feedback,
            "session_id": request.session_id,
            "message": "Feedback processed, agent response is being updated"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/agentic-ai/status/{session_id}")
async def get_workflow_status(session_id: str):
    """Get the current status of a workflow"""
    try:
        session = active_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Calculate overall progress
        total_agents = len(session["agents"])
        completed_agents = len([k for k, v in session["completion_status"].items() if v and k.startswith("agent_")])
        progress_percentage = (completed_agents / total_agents * 100) if total_agents > 0 else 0
        
        return {
            "session_id": session_id,
            "status": session.get("status", "active"),
            "current_agent_index": session.get("current_agent_index", 0),
            "total_agents": total_agents,
            "completed_agents": completed_agents,
            "progress_percentage": round(progress_percentage, 1),
            "completion_status": session["completion_status"],
            "agent_responses": session["agent_responses"],
            "conversation_history": session["conversation_history"],
            "feedback_history": session.get("feedback_history", {}),
            "pre_agent_instructions": session.get("pre_agent_instructions", {}),  # NEW: Include pre-agent instructions
            "created_at": session["created_at"],
            "agents": [{"name": agent["name"], "index": i} for i, agent in enumerate(session["agents"])]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/agentic-ai/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session"""
    try:
        if session_id in active_sessions:
            del active_sessions[session_id]
            return {"status": "deleted", "session_id": session_id}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/agentic-ai/sessions")
async def list_active_sessions():
    """List all active sessions"""
    try:
        sessions_summary = []
        for session_id, session_data in active_sessions.items():
            sessions_summary.append({
                "session_id": session_id,
                "status": session_data.get("status", "unknown"),
                "current_agent_index": session_data.get("current_agent_index", 0),
                "total_agents": len(session_data.get("agents", [])),
                "created_at": session_data.get("created_at"),
                "has_pre_instructions": len(session_data.get("pre_agent_instructions", {})) > 0  # NEW
            })
        
        return {
            "active_sessions": sessions_summary,
            "total_sessions": len(active_sessions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# NEW: Endpoint to get pre-agent instructions for a specific agent
@app.get("/api/v1/agentic-ai/session/{session_id}/pre-instructions/{agent_index}")
async def get_pre_agent_instructions(session_id: str, agent_index: int):
    """Get pre-agent instructions for a specific agent"""
    try:
        session = active_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        instructions = session.get("pre_agent_instructions", {}).get(agent_index)
        
        return {
            "session_id": session_id,
            "agent_index": agent_index,
            "pre_agent_instructions": instructions,
            "has_instructions": instructions is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Memory v2 endpoints (enhanced with pre-agent instructions support)
@app.post("/api/v1/memory/store")
async def store_memory(data: Dict[str, Any]):
    """Store data in memory v2"""
    try:
        key = data.get("key")
        value = data.get("value")
        
        if not key:
            raise HTTPException(status_code=400, detail="Key is required")
        
        memory_v2_data[key] = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "type": type(value).__name__
        }
        
        return {"status": "stored", "key": key}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/memory/retrieve/{key}")
async def retrieve_memory(key: str):
    """Retrieve data from memory v2"""
    try:
        if key in memory_v2_data:
            return memory_v2_data[key]
        else:
            raise HTTPException(status_code=404, detail="Key not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/memory/list")
async def list_memory_keys():
    """List all memory keys"""
    try:
        return {
            "keys": list(memory_v2_data.keys()),
            "total_keys": len(memory_v2_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
