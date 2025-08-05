from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
from typing import AsyncGenerator, Optional
import json
import uuid
from langgraph_agent import LangGraphAgent

app = FastAPI(title="LangGraph vLLM Chatbot API", description="Streaming chatbot using LangGraph and vLLM with session memory")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],  # In production, replace "*" with your specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods including OPTIONS, GET, POST
    allow_headers=["*"],  # Allows all headers
)

class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful AI assistant. Provide clear and concise responses."
    session_id: Optional[str] = None

# Initialize the LangGraph agent
chat_agent = LangGraphAgent()

@app.get("/")
async def root():
    return {"message": "LangGraph vLLM Chatbot API is running"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    async def generate_response():
        # Send start signal
        yield "data: {\"type\": \"start\"}\n\n"
        
        # Stream the agent response
        async for chunk in chat_agent.stream_response(request.message, session_id):
            yield chunk
            await asyncio.sleep(0.01)
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "agent": "ready"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)