"""Main FastAPI application"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging

from docs_logic import DocumentationService
from docs_router import router as docs_router

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Documentation API")

# CORS configuration
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize documentation service
DOCS_DIR = Path("../docs")
INDEX_DIR = Path("./search_index")

# Create a single instance of the service
doc_service = DocumentationService(docs_dir=DOCS_DIR, index_dir=INDEX_DIR)

def get_documentation_service() -> DocumentationService:
    """Dependency to provide documentation service to routers"""
    return doc_service

# Include routers
app.include_router(docs_router)

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Documentation API is running"}

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize search index on startup"""
    logger.info("Starting up Documentation API...")
    doc_service.initialize_or_recreate_index()
    doc_service.index_documents()
    logger.info("Documentation API started successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down Documentation API...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)