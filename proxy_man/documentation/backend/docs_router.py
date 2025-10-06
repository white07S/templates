"""API Router for documentation endpoints"""
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import aiofiles
import logging

from docs_logic import DocumentationService

# Set up logging
logger = logging.getLogger(__name__)

# Pydantic models
class Page(BaseModel):
    id: str
    title: str
    path: str
    content: Optional[str] = None

class PageContent(BaseModel):
    id: str
    title: str
    content: str
    path: str

class SearchResult(BaseModel):
    id: str
    title: str
    path: str
    score: float
    snippet: str
    category: str  # 'heading', 'content', 'code'
    heading: Optional[str] = None  # The heading this content belongs to
    anchor: Optional[str] = None  # Anchor to jump to specific section

class NavigationItem(BaseModel):
    id: str
    title: str
    path: str
    children: Optional[List['NavigationItem']] = []

NavigationItem.model_rebuild()

# Create router
router = APIRouter(prefix="/api", tags=["documentation"])

# Dependency to get the documentation service
def get_doc_service() -> DocumentationService:
    """Get documentation service instance from app state"""
    from main import get_documentation_service
    return get_documentation_service()

@router.get("/pages", response_model=List[NavigationItem])
async def get_pages(service: DocumentationService = Depends(get_doc_service)):
    """Get all pages for navigation"""
    pages = service.get_navigation_structure()
    # Convert dict to NavigationItem
    return [NavigationItem(**page) for page in pages]

@router.get("/page/{page_id}", response_model=PageContent)
async def get_page(page_id: str, service: DocumentationService = Depends(get_doc_service)):
    """Get specific page content"""
    page_content = service.get_page_content(page_id)

    if not page_content:
        raise HTTPException(status_code=404, detail="Page not found")

    return PageContent(**page_content)

@router.get("/search", response_model=List[SearchResult])
async def search_pages(
    q: str = Query(..., min_length=1),
    category: Optional[str] = None,
    limit: int = Query(default=20, le=100),
    service: DocumentationService = Depends(get_doc_service)
):
    """Enhanced search with categories and snippets"""
    try:
        results = service.search_documents(q, category, limit)
        return [SearchResult(**result) for result in results]
    except Exception as e:
        if "not initialized" in str(e):
            raise HTTPException(status_code=503, detail="Search index not initialized")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/image/{image_path:path}")
async def get_image(image_path: str, service: DocumentationService = Depends(get_doc_service)):
    """Serve images from the docs/images directory"""
    full_path = service.images_dir / image_path

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(full_path)

@router.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    service: DocumentationService = Depends(get_doc_service)
):
    """Upload an image to the docs/images directory"""
    # Validate file type
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Save file
    file_path = service.images_dir / file.filename

    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)

    return {"filename": file.filename, "path": f"images/{file.filename}"}

@router.post("/reindex")
async def reindex_search(service: DocumentationService = Depends(get_doc_service)):
    """Manually trigger search index rebuild"""
    try:
        service.index_documents()
        return {"message": "Search index rebuilt successfully"}
    except Exception as e:
        logger.error(f"Reindex failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reindex failed: {str(e)}")

@router.get("/navigation/{page_id}")
async def get_navigation_context(page_id: str, service: DocumentationService = Depends(get_doc_service)):
    """Get previous and next page for navigation"""
    context = service.get_navigation_context(page_id)

    if not context:
        raise HTTPException(status_code=404, detail="Page not found")

    return context

@router.get("/health")
async def health_check(service: DocumentationService = Depends(get_doc_service)):
    """Check the health of the API and search index"""
    return service.get_health_status()