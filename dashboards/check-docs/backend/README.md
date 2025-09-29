# Documentation Backend API

## Setup Instructions

### 1. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Run the FastAPI Server

```bash
# From the backend directory
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /` - API info and version
- `GET /api/docs` - List all documentation
- `GET /api/docs/{doc_id}` - Get specific document content
- `GET /api/docs/{doc_id}/raw` - Get raw MDX content
- `GET /api/search?q={query}` - Search documentation

## Frontend Setup

### 1. Install npm dependencies

```bash
# From the project root
npm install
```

### 2. Start the React development server

```bash
npm start
```

The frontend will be available at `http://localhost:3000`

## Environment Variables

Create a `.env` file in the React project root:

```env
REACT_APP_API_URL=http://localhost:8000
```

## Running Both Services

### Terminal 1 - Backend:
```bash
cd backend
python main.py
```

### Terminal 2 - Frontend:
```bash
npm start
```

## Architecture

- **Backend (FastAPI)**: Serves documentation files as a REST API
- **Frontend (React)**: Fetches and renders documentation from the API
- **MDX Processing**: Documents are processed on the frontend using @mdx-js/mdx

## Benefits

1. **Separation of Concerns**: Backend handles file serving, frontend handles rendering
2. **Scalability**: Backend can be deployed separately and scaled independently
3. **Caching**: API responses can be cached for better performance
4. **Security**: Better control over which files are served
5. **Dynamic Content**: Documents can be updated without rebuilding the frontend