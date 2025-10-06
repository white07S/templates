# Documentation Backend with Tantivy Search

A FastAPI-based documentation backend with fast and accurate full-text search powered by Tantivy (embedded Rust search engine).

## Features

- Fast embedded full-text search with Tantivy (no external server required)
- MDX document support
- Granular section-level search (headings, content, code blocks)
- Category-based filtering
- Snippet generation
- Automatic schema versioning and migration
- Image upload and serving
- Navigation structure generation
- CORS support for frontend integration

## Prerequisites

- Python 3.12+
- No external services required (Tantivy is embedded)

## Installation

### Install Python Dependencies

Using uv (recommended):
```bash
uv pip install -e .
```

Or using pip:
```bash
pip install -e .
```

### Run the Backend

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## Key Features

### Automatic Schema Management
The system automatically handles schema changes:
- Tracks schema version in `search_index/schema_version.json`
- Automatically rebuilds index when schema changes
- No manual intervention needed for schema updates

### Search Index Structure
- **Location**: `./search_index/` directory
- **Schema Version**: Tracked automatically
- **Rebuild**: Automatic on schema changes or via API endpoint

## API Endpoints

### Core Endpoints
- `GET /`: Health check
- `GET /api/health`: Detailed health status including index info

### Navigation
- `GET /api/pages`: Get all pages for navigation
- `GET /api/page/{page_id}`: Get specific page content
- `GET /api/navigation/{page_id}`: Get previous/next navigation context

### Search
- `GET /api/search?q={query}`: Search documents
  - Optional: `category={heading|content|code}` - Filter by content type
  - Optional: `limit={number}` - Limit results (max 100)

### Images
- `GET /api/image/{image_path}`: Serve images
- `POST /api/upload-image`: Upload images

### Maintenance
- `POST /api/reindex`: Manually trigger search index rebuild

## Search Features

### Granular Search
The search system indexes documents at multiple levels:
- **Headings**: Indexed separately for direct navigation
- **Content**: Regular text content under each heading
- **Code Blocks**: Separately indexed code snippets

### Tokenizers
Different field types use appropriate tokenizers:
- **Text fields**: English stemming tokenizer for better search
- **IDs/Paths/Anchors**: Raw tokenizer for exact matching
- **Categories**: Raw tokenizer for exact filtering

## Document Structure

Place your MDX files in the `../docs` directory. The system will:
1. Automatically discover all `.mdx` files
2. Extract titles from H1 headings or filenames
3. Create navigation structure
4. Index content for search

Example structure:
```
docs/
├── getting-started.mdx
├── api/
│   ├── authentication.mdx
│   └── endpoints.mdx
└── images/
    └── logo.png
```

## Frontend Integration

Example frontend request:
```javascript
// Search
const results = await fetch('http://localhost:8000/api/search?q=authentication')
  .then(res => res.json());

// Get page content
const page = await fetch('http://localhost:8000/api/page/getting-started')
  .then(res => res.json());
```

## Troubleshooting

### Schema Mismatch Error
The system automatically handles schema mismatches by:
1. Detecting version changes
2. Recreating the index with the new schema
3. Re-indexing all documents

To force a rebuild:
```bash
rm -rf search_index
python main.py
```

Or via API:
```bash
curl -X POST http://localhost:8000/api/reindex
```

### Search Not Working
1. Check API health: `curl http://localhost:8000/api/health`
2. Verify index exists: Check if `search_index/` directory exists
3. Manually reindex: `curl -X POST http://localhost:8000/api/reindex`
4. Check logs for indexing errors

### Index Corruption
If the index becomes corrupted:
```bash
rm -rf search_index
python main.py  # Will automatically recreate
```

## Testing

Run the test suite:
```bash
python test_api.py
```

## Performance

Tantivy provides:
- Fast in-memory search
- No network overhead (embedded)
- Efficient disk storage
- Automatic index optimization
- Multi-threaded indexing

## Development

### Changing the Schema
1. Modify the `create_schema()` function in `main.py`
2. Increment `CURRENT_SCHEMA_VERSION`
3. Restart the server - it will automatically rebuild

### Docker Support

Build and run with Docker:
```bash
docker build -t docs-backend .
docker run -p 8000:8000 -v $(pwd)/../docs:/docs docs-backend
```

## Advantages of Tantivy

- **No External Dependencies**: Runs embedded, no separate server needed
- **Fast**: Written in Rust for maximum performance
- **Reliable**: No network issues or service management
- **Simple Deployment**: Just Python, no infrastructure needed
- **Low Resource Usage**: Efficient memory and disk usage

## License

[Your License Here]