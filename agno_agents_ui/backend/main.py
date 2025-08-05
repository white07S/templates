from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
from typing import AsyncGenerator
import json
import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat




app = FastAPI(title="Agno Chatbot API", description="Streaming chatbot using Agno and OpenAI GPT-4o-mini")

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

class ChatAgent:
    def __init__(self):
        self.agent = Agent(
            model=OpenAIChat(
                id="gpt-4o-mini",
                api_key=os.getenv("OPENAI_API_KEY")
            ),
            instructions="You are a helpful AI assistant. Provide clear, accurate, and engaging responses.",
            markdown=True,
            stream=True
        )
    
    async def stream_response(self, message: str, system_prompt: str = None) -> AsyncGenerator[str, None]:
        try:
            if system_prompt:
                self.agent.instructions = system_prompt
            
            response = self.agent.run(message, stream=True)
            
            if hasattr(response, '__iter__'):
                for chunk in response:
                    if hasattr(chunk, 'content') and chunk.content:
                        yield f"data: {json.dumps({'content': chunk.content, 'type': 'content'})}\n\n"
                    elif isinstance(chunk, str) and chunk:
                        yield f"data: {json.dumps({'content': chunk, 'type': 'content'})}\n\n"
            else:
                yield f"data: {json.dumps({'content': str(response), 'type': 'content'})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'type': 'error'})}\n\n"

chat_agent = ChatAgent()

@app.get("/")
async def root():
    return {"message": "Agno Chatbot API is running"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    async def generate_response():
        yield "data: {\"type\": \"start\"}\n\n"
        async for chunk in chat_agent.stream_response(request.message, request.system_prompt):
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