# LangGraph + vLLM Backend

This backend provides a FastAPI server with streaming support for LangGraph agents powered by vLLM, including session memory for follow-up conversations.

## Features

- **Streaming responses** via Server-Sent Events (SSE)
- **Session memory** for contextual follow-up questions
- **Tool calling** support with weather, calculator, and web search tools
- **WebSocket support** for bidirectional streaming
- **vLLM integration** for fast inference

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
# or with uv:
uv pip install -e .
```

2. Configure environment variables:
```bash
export VLLM_BASE_URL="http://localhost:8000/v1"  # Your vLLM server URL
export VLLM_API_KEY="EMPTY"                       # vLLM doesn't need API key
export MODEL_NAME="your-model-name"              # Your model name
```

3. Start the server:
```bash
uvicorn main:app --reload --port 8000
```

## API Endpoints

### Chat Endpoint
- **POST** `/chat` - Send a message and receive streaming response
  ```json
  {
    "message": "What's the weather in New York?",
    "session_id": "optional-session-id"
  }
  ```

### Session Management
- **GET** `/session/{session_id}` - Get session information
- **DELETE** `/session/{session_id}` - Clear session history
- **POST** `/session/new` - Create a new session

### WebSocket
- **WS** `/ws/{session_id}` - WebSocket connection for bidirectional streaming

### Health Check
- **GET** `/health` - Check server status

## Response Format

The streaming response includes different event types:
- `session` - Session ID information
- `content` - LLM text content
- `tool_call` - Tool invocation details
- `tool_result` - Tool execution results
- `done` - Stream completion
- `error` - Error messages