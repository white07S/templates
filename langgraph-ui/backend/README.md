# LangGraph OpenAI Streaming API

A FastAPI application that provides streaming chat capabilities using OpenAI's o3-mini model with LangGraph for tool integration.

## Features

- **Streaming Responses**: Real-time streaming of AI responses
- **Tool Integration**: Built-in tools for weather, calculations, and web search
- **Session Management**: Persistent conversation sessions
- **Multiple Endpoints**: Different streaming modes available

## Setup

1. **Install Dependencies**:
```bash
pip install -e .
```

2. **Configure Environment**:
```bash
cp .env.example .env
```
Then edit `.env` with your OpenAI API key:
```
OPENAI_API_KEY=your_actual_api_key_here
```

3. **Run the Server**:
```bash
python app.py
```

The server will start on `http://0.0.0.0:8080`

## Endpoints

### Streaming Chat with Tools
- **POST** `/chat/stream` - Full LangGraph streaming with tool support
- **POST** `/chat/direct-stream` - Direct OpenAI streaming (no tools)

### Non-streaming Chat
- **POST** `/chat` - Standard chat endpoint

### Testing
- **POST** `/test/openai-stream` - Test OpenAI streaming functionality

### Session Management
- **GET** `/session/{session_id}/history` - Get session history
- **DELETE** `/session/{session_id}` - Clear session

### Health Check
- **GET** `/health` - API health status

## Environment Variables

- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `MODEL_NAME` - Model to use (default: "o3-mini")
- `TEMPERATURE` - Sampling temperature (default: 0.7)
- `MAX_RETRIES` - Request retry limit (default: 3)
- `TIMEOUT` - Request timeout in seconds (default: 30)
- `STREAM_DELAY` - Streaming delay between chunks (default: 0.001)

## Built-in Tools

1. **Weather Tool**: Get weather information for various cities
2. **Calculator**: Perform mathematical calculations
3. **Web Search**: Search for information (simulated responses)

## Usage Example

```python
import requests
import json

# Streaming request
response = requests.post(
    "http://localhost:8080/chat/stream",
    json={
        "message": "What's the weather in New York and calculate 15 * 23?",
        "session_id": "test-session",
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line.startswith(b"data: "):
        data = json.loads(line[6:])
        if data.get("type") == "content":
            print(data.get("data"), end="", flush=True)
```