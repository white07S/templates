# Quick Start Guide

## One-API Deployment

Start One-API using Docker:

```bash
docker run --name one-api -d --restart always \
  -p 13000:3000 \
  -e TZ=Asia/Shanghai \
  -v /home/ubuntu/data/one-api:/data \
  justsong/one-api
```

Configure third-party API Keys in the One-API console. All API requests from this project will be forwarded through One-API.

The official address of this project: https://github.com/songquanpeng/one-api

For specific configuration instructions, see [here](https://github.com/1517005260/graph-rag-agent/issues/7#issuecomment-2906770240)

**Note**: By default, log in with the administrator account, username: root, password: 123456. You can change the password after logging in.

## Neo4j Startup

```bash
cd graph-rag-agent/
docker compose up -d
```

Default account credentials:

```
Username: neo4j
Password: 12345678
```

## Environment Setup

```bash
conda create -n graphrag python==3.10
conda activate graphrag
cd graph-rag-agent/
pip install -r requirements.txt
```

Note: To process `.doc` format (old version Word files), please install the appropriate dependencies according to your operating system, as detailed in the comments in `requirements.txt`:

```txt
# Linux
sudo apt-get install python-dev-is-python3 libxml2-dev libxslt1-dev antiword unrtf poppler-utils

# Windows
pywin32>=302

textract==1.6.3  # No need to install on Windows
```

## .env Configuration

Create a `.env` file in the project root directory, example:

```env
OPENAI_API_KEY = 'sk-xxx'  # api-key
OPENAI_BASE_URL = 'http://localhost:13000/v1' # url

OPENAI_EMBEDDINGS_MODEL = 'text-embedding-3-large'  # Vector embedding model
OPENAI_LLM_MODEL = 'gpt-4o'  # Dialog model

TEMPERATURE = 0   # Model divergence, 0-1, higher values produce more creative answers
MAX_TOKENS = 2000  # Maximum tokens

VERBOSE = True  # Debug mode

# neo4j configuration
NEO4J_URI='neo4j://localhost:7687'
NEO4J_USERNAME='neo4j'
NEO4J_PASSWORD='12345678'

# Cache vector similarity matching configuration
# Optional values: 'openai' (reuse RAG vector model), 'sentence_transformer' (use local model)
CACHE_EMBEDDING_PROVIDER = 'openai'
# Model name when using sentence_transformer, model will be cached to ./cache/model directory
CACHE_SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'
# Model cache configuration
MODEL_CACHE_ROOT = './cache'  # Cache root directory, models will be saved to {MODEL_CACHE_ROOT}/model

# LangSmith configuration (optional, can be commented out if monitoring is not needed)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="xxx"
LANGSMITH_PROJECT="xxx"
```

**Note**: Only deepseek (20241226 version) and gpt-4o have been fully tested. Other models, such as deepseek (20250324 version), have more serious hallucination problems and may not follow prompts properly, leading to entity extraction failures. Qwen models can extract entities but seem not to support langchain/langgraph, so there's a chance of errors during Q&A. They have their own agent implementation [Qwen-Agent](https://qwen.readthedocs.io/zh-cn/latest/framework/qwen_agent.html).

## Project Initialization

```bash
pip install -e .
```

## Knowledge Graph Source File Placement

Please place source files in the `files/` folder, supporting hierarchical directory storage. Currently supports the following formats (using simple chunking, processing methods will be optimized later):

```
- TXT (plain text)
- PDF (PDF documents)
- MD (Markdown)
- DOCX (new Word documents)
- DOC (old Word documents)
- CSV (spreadsheets)
- JSON (structured text)
- YAML/YML (configuration files)
```

## Knowledge Graph Configuration (`config/settings.py`)

```python
# Basic settings
theme = "Journey to the West"
entity_types = ["character", "demon", "location"]
relationship_types = ["master-disciple", "brotherhood", "antagonist", "dialogue", "attitude", "story_location", "other"]

# Incremental update settings: conflict resolution strategy (conflicts between new files and manual Neo4j edits), can be "manual_first" (prioritize manual edits), "auto_first" (prioritize automatic updates) or "merge" (attempt to merge)
conflict_strategy="manual_first"

# Graph parameters
similarity_threshold = 0.9
community_algorithm = 'leiden'  # Options: sllpa or leiden

# Text chunking parameters
CHUNK_SIZE = 300
OVERLAP = 50
MAX_TEXT_LENGTH = 500000 # Maximum processing length

# Response type
response_type = "multiple paragraphs"

# Agent tool descriptions
lc_description = "For queries requiring specific details, such as dialogues and scene descriptions in 'Journey to the West'."
gl_description = "For macro summaries and analysis, such as character relationships and theme development."
naive_description = "Basic retrieval tool that returns the most relevant original text passages."

# Performance optimization parameters
# Parallel processing configuration
MAX_WORKERS = 4                # Number of parallel worker threads
BATCH_SIZE = 100               # Batch processing size
ENTITY_BATCH_SIZE = 50         # Entity processing batch size
CHUNK_BATCH_SIZE = 100         # Text chunk processing batch size
EMBEDDING_BATCH_SIZE = 64      # Embedding vector computation batch size
LLM_BATCH_SIZE = 5             # LLM processing batch size

# GDS related configuration
GDS_MEMORY_LIMIT = 6           # GDS memory limit (GB)
GDS_CONCURRENCY = 4            # GDS concurrency
GDS_NODE_COUNT_LIMIT = 50000   # GDS node count limit
GDS_TIMEOUT_SECONDS = 300      # GDS timeout (seconds)

# Index and community detection configuration
COMMUNITY_BATCH_SIZE = 50      # Community processing batch size
```

## Building Knowledge Graph

```bash
cd graph-rag-agent/

# Initial full build
python build/main.py

# Single incremental (increment/decrement) build:
python build/incremental_update.py --once

# Background daemon, periodic incremental updates:
python build/incremental_update.py --daemon
```

**Note:** `main.py` is the full construction process. If you need to run individual processes separately, please complete entity index construction first, then proceed with chunk index construction, otherwise errors will occur (chunk index depends on entity index).

## Knowledge Graph Search Testing

```bash
cd graph-rag-agent/test

# You can comment out Agents you don't want to test before querying to prevent slow execution

# Non-streaming query
python search_without_stream.py

# Streaming query
python search_with_stream.py
```

## Knowledge Graph Evaluation

```bash
cd evaluator/test
# See corresponding README for more information
```

## Example Question Configuration (for frontend display)

Edit the `examples` field in `config/settings.py`:

```python
examples = [
    "Who are the main characters in 'Journey to the West'?",
    "What did Tang Monk discuss with the talking tree?",
    "What story is there between Sun Wukong and the female demon?",
    "What was his final choice?"
]
```

## Concurrent Process Configuration (`server/main.py`)

```python
# FastAPI concurrent process number setting
workers = 2
```

## Deep Search Optimization (recommended to disable frontend timeout)

If you need to enable deep search functionality, it's recommended to disable frontend timeout limits, modify `frontend/utils/api.py`:

```python
response = requests.post(
    f"{API_URL}/chat",
    json={
        "message": message,
        "session_id": st.session_state.session_id,
        "debug": st.session_state.debug_mode,
        "agent_type": st.session_state.agent_type
    },
    # timeout=120  # Recommend commenting out this line
)
```

## Chinese Font Support (Linux)

For Chinese chart display, refer to [font installation tutorial](https://zhuanlan.zhihu.com/p/571610437). By default, English plotting is used (`matplotlib`).

## Starting Frontend and Backend Services

```bash
# Start backend
cd graph-rag-agent/
python server/main.py

# Start frontend
cd graph-rag-agent/
streamlit run frontend/app.py
```

**Note**: Due to langchain version issues, the current streaming is pseudo-streaming implementation, i.e., the complete answer is generated first, then returned in segments.