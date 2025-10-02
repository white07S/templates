from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import tantivy
from pathlib import Path
import aiofiles
import shutil
import hashlib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Base directory for docs
DOCS_DIR = Path("../docs")
DOCS_DIR.mkdir(exist_ok=True)
IMAGES_DIR = DOCS_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# Tantivy search index setup
INDEX_DIR = Path("./search_index")
SCHEMA_VERSION_FILE = INDEX_DIR / "schema_version.json"
CURRENT_SCHEMA_VERSION = "2.0.0"  # Increment this when schema changes

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

# Global index variable
index = None
searcher = None

def create_schema():
    """Create the Tantivy schema"""
    schema_builder = tantivy.SchemaBuilder()

    # Document fields with proper configuration
    schema_builder.add_text_field("id", stored=True, tokenizer_name="raw")  # Use raw tokenizer for IDs
    schema_builder.add_text_field("title", stored=True, tokenizer_name="en_stem")
    schema_builder.add_text_field("content", stored=True, tokenizer_name="en_stem")
    schema_builder.add_text_field("path", stored=True, tokenizer_name="raw")  # Use raw tokenizer for paths
    schema_builder.add_text_field("category", stored=True, tokenizer_name="raw")  # Categories don't need stemming
    schema_builder.add_text_field("heading", stored=True, tokenizer_name="en_stem")
    schema_builder.add_text_field("anchor", stored=True, tokenizer_name="raw")  # Anchors are exact strings
    schema_builder.add_text_field("doc_title", stored=True, tokenizer_name="en_stem")

    return schema_builder.build()

def check_schema_compatibility():
    """Check if the existing index schema is compatible"""
    if not INDEX_DIR.exists():
        return False

    if not SCHEMA_VERSION_FILE.exists():
        return False

    try:
        with open(SCHEMA_VERSION_FILE, 'r') as f:
            version_info = json.load(f)
            return version_info.get('version') == CURRENT_SCHEMA_VERSION
    except:
        return False

def save_schema_version():
    """Save the current schema version"""
    INDEX_DIR.mkdir(exist_ok=True)
    with open(SCHEMA_VERSION_FILE, 'w') as f:
        json.dump({
            'version': CURRENT_SCHEMA_VERSION,
            'fields': [
                'id', 'title', 'content', 'path',
                'category', 'heading', 'anchor', 'doc_title'
            ]
        }, f, indent=2)

def initialize_or_recreate_index():
    """Initialize the index, recreating if schema has changed"""
    global index, searcher

    schema = create_schema()

    # Check if we need to recreate the index
    if INDEX_DIR.exists():
        if not check_schema_compatibility():
            logger.info("Schema version mismatch. Recreating index...")
            shutil.rmtree(INDEX_DIR)
            INDEX_DIR.mkdir(exist_ok=True)
        else:
            try:
                # Try to open existing index
                index = tantivy.Index(str(INDEX_DIR))
                searcher = index.searcher()
                logger.info("Loaded existing Tantivy index")
                return
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Recreating...")
                shutil.rmtree(INDEX_DIR)
                INDEX_DIR.mkdir(exist_ok=True)
    else:
        INDEX_DIR.mkdir(exist_ok=True)

    # Create new index
    index = tantivy.Index(schema, str(INDEX_DIR))
    searcher = index.searcher()
    save_schema_version()
    logger.info("Created new Tantivy index")

# Helper functions
def extract_sections_from_mdx(content: str, doc_id: str, path: str):
    """Extract different sections from MDX content for granular indexing"""
    sections = []
    lines = content.split("\n")
    current_heading = None
    current_content = []
    current_heading_level = 0

    # Extract document title
    doc_title = path.replace(".mdx", "").replace("-", " ").title()
    for line in lines:
        if line.startswith("# "):
            doc_title = line[2:].strip()
            break

    for i, line in enumerate(lines):
        # Check if it's a heading
        if line.startswith("#"):
            # Save previous section if exists
            if current_content and current_heading:
                content_text = "\n".join(current_content).strip()
                if content_text:
                    # Create anchor from heading
                    anchor = current_heading.lower().replace(" ", "-").replace(".", "")
                    sections.append({
                        "id": f"{doc_id}#{anchor}",
                        "title": doc_title,
                        "content": content_text,
                        "heading": current_heading,
                        "category": "content",
                        "anchor": anchor,
                        "path": path,
                        "doc_title": doc_title
                    })

            # Extract heading level and text
            heading_match = line.split(" ", 1)
            if len(heading_match) > 1:
                current_heading_level = len(heading_match[0])
                current_heading = heading_match[1].strip()
                current_content = []

                # Index the heading itself
                anchor = current_heading.lower().replace(" ", "-").replace(".", "")
                sections.append({
                    "id": f"{doc_id}#{anchor}-heading",
                    "title": doc_title,
                    "content": current_heading,
                    "heading": current_heading,
                    "category": "heading",
                    "anchor": anchor,
                    "path": path,
                    "doc_title": doc_title
                })

        # Check if it's a code block
        elif line.startswith("```"):
            code_lines = []
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                code_lines.append(lines[j])
                j += 1

            if code_lines:
                code_content = "\n".join(code_lines)
                # Create a stable anchor for code blocks
                code_hash = hashlib.md5(code_content.encode()).hexdigest()[:8]
                anchor = f"code-{code_hash}"
                sections.append({
                    "id": f"{doc_id}#{anchor}",
                    "title": doc_title,
                    "content": code_content,
                    "heading": current_heading or "",
                    "category": "code",
                    "anchor": anchor,
                    "path": path,
                    "doc_title": doc_title
                })
        else:
            current_content.append(line)

    # Don't forget the last section
    if current_content and current_heading:
        content_text = "\n".join(current_content).strip()
        if content_text:
            anchor = current_heading.lower().replace(" ", "-").replace(".", "")
            sections.append({
                "id": f"{doc_id}#{anchor}",
                "title": doc_title,
                "content": content_text,
                "heading": current_heading,
                "category": "content",
                "anchor": anchor,
                "path": path,
                "doc_title": doc_title
            })

    return sections

def index_documents():
    """Index all MDX documents for search with granular sections"""
    global index, searcher

    if not index:
        initialize_or_recreate_index()

    writer = index.writer()

    # Clear existing documents
    writer.delete_all_documents()

    document_count = 0

    # Index all MDX files
    for mdx_file in DOCS_DIR.glob("**/*.mdx"):
        relative_path = str(mdx_file.relative_to(DOCS_DIR))
        doc_id = relative_path.replace("/", "_").replace(".mdx", "")

        try:
            with open(mdx_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract sections for granular indexing
            sections = extract_sections_from_mdx(content, doc_id, relative_path)

            # Index each section separately
            for section in sections:
                # Combine heading and content for better search
                search_content = section["content"]
                if section.get("heading") and section["category"] != "heading":
                    search_content = f"{section['heading']} {search_content}"

                doc = tantivy.Document()
                doc.add_text("id", section["id"])
                doc.add_text("title", section["title"])
                doc.add_text("content", search_content)
                doc.add_text("path", section["path"])
                doc.add_text("category", section["category"])
                doc.add_text("heading", section.get("heading", ""))
                doc.add_text("anchor", section.get("anchor", ""))
                doc.add_text("doc_title", section["doc_title"])

                writer.add_document(doc)
                document_count += 1

        except Exception as e:
            logger.error(f"Error indexing {mdx_file}: {e}")
            continue

    writer.commit()
    index.reload()
    searcher = index.searcher()

    logger.info(f"Successfully indexed {document_count} document sections")

def get_navigation_structure() -> List[NavigationItem]:
    """Build navigation structure from file system"""
    nav_items = []

    # Get all MDX files and organize by directory
    for mdx_file in sorted(DOCS_DIR.glob("**/*.mdx")):
        relative_path = mdx_file.relative_to(DOCS_DIR)
        parts = relative_path.parts

        # Extract title
        with open(mdx_file, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")
            title = mdx_file.stem.replace("-", " ").title()
            for line in lines:
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

        doc_id = str(relative_path).replace("/", "_").replace(".mdx", "")

        # Create navigation item
        nav_item = NavigationItem(
            id=doc_id,
            title=title,
            path=str(relative_path)
        )

        # For simplicity, we'll just add all items to root level
        # You can extend this to create nested structure based on directories
        nav_items.append(nav_item)

    return nav_items

def create_snippet(content: str, query: str, max_length: int = 150) -> str:
    """Create a snippet with the query highlighted"""
    content_lower = content.lower()
    query_lower = query.lower()

    # Find the position of the query in the content
    pos = content_lower.find(query_lower)

    if pos == -1:
        # If exact match not found, try to find any of the words
        words = query_lower.split()
        for word in words:
            pos = content_lower.find(word)
            if pos != -1:
                break

    if pos == -1:
        # Return beginning of content if no match
        return content[:max_length] + "..." if len(content) > max_length else content

    # Calculate snippet boundaries
    start = max(0, pos - 50)
    end = min(len(content), pos + len(query) + 100)

    snippet = content[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."

    return snippet

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize search index on startup"""
    initialize_or_recreate_index()
    index_documents()

@app.get("/")
async def root():
    return {"message": "Documentation API is running"}

@app.get("/api/pages", response_model=List[NavigationItem])
async def get_pages():
    """Get all pages for navigation"""
    return get_navigation_structure()

@app.get("/api/page/{page_id}", response_model=PageContent)
async def get_page(page_id: str):
    """Get specific page content"""
    # Convert ID back to file path
    file_path = page_id.replace("_", "/") + ".mdx"
    full_path = DOCS_DIR / file_path

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Page not found")

    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract title
    lines = content.split("\n")
    title = full_path.stem.replace("-", " ").title()
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break

    return PageContent(
        id=page_id,
        title=title,
        content=content,
        path=file_path
    )

@app.get("/api/search", response_model=List[SearchResult])
async def search_pages(
    q: str = Query(..., min_length=1),
    category: Optional[str] = None,
    limit: int = Query(default=20, le=100)
):
    """Enhanced search with categories and snippets"""
    global searcher

    if not searcher:
        raise HTTPException(status_code=503, detail="Search index not initialized")

    # Build query - search across multiple fields
    query_parser = index.parse_query(q, ["title", "content", "heading", "doc_title"])

    search_results = searcher.search(query_parser, limit=limit * 2)  # Get more results for filtering

    results = []
    seen_ids = set()

    for score, doc_address in search_results.hits:
        doc = searcher.doc(doc_address)

        doc_id = doc.get_first("id")
        if not doc_id:
            continue

        # Skip duplicates
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)

        # Get all fields with fallbacks
        title = doc.get_first("doc_title") or doc.get_first("title") or "Untitled"
        content = doc.get_first("content") or ""
        path = doc.get_first("path") or ""
        result_category = doc.get_first("category") or "content"
        heading = doc.get_first("heading") or ""
        anchor = doc.get_first("anchor") or ""

        # Filter by category if specified
        if category and result_category != category:
            continue

        # Create snippet
        snippet = create_snippet(content, q)

        results.append(SearchResult(
            id=doc_id,
            title=title,
            path=path,
            score=score,
            snippet=snippet,
            category=result_category,
            heading=heading if heading else None,
            anchor=anchor if anchor else None
        ))

        # Stop if we have enough results
        if len(results) >= limit:
            break

    # Sort by score (highest first)
    results.sort(key=lambda x: x.score, reverse=True)

    return results

@app.get("/api/image/{image_path:path}")
async def get_image(image_path: str):
    """Serve images from the docs/images directory"""
    full_path = IMAGES_DIR / image_path

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(full_path)

@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image to the docs/images directory"""
    # Validate file type
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Save file
    file_path = IMAGES_DIR / file.filename

    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)

    return {"filename": file.filename, "path": f"images/{file.filename}"}

@app.post("/api/reindex")
async def reindex_search():
    """Manually trigger search index rebuild"""
    try:
        index_documents()
        return {"message": "Search index rebuilt successfully"}
    except Exception as e:
        logger.error(f"Reindex failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reindex failed: {str(e)}")

@app.get("/api/navigation/{page_id}")
async def get_navigation_context(page_id: str):
    """Get previous and next page for navigation"""
    pages = get_navigation_structure()

    # Find current page index
    current_index = -1
    for i, page in enumerate(pages):
        if page.id == page_id:
            current_index = i
            break

    if current_index == -1:
        raise HTTPException(status_code=404, detail="Page not found")

    previous_page = pages[current_index - 1] if current_index > 0 else None
    next_page = pages[current_index + 1] if current_index < len(pages) - 1 else None

    return {
        "previous": previous_page,
        "next": next_page,
        "current": pages[current_index]
    }

@app.get("/api/health")
async def health_check():
    """Check the health of the API and search index"""
    try:
        index_exists = INDEX_DIR.exists()
        schema_version = None
        doc_count = 0

        if SCHEMA_VERSION_FILE.exists():
            with open(SCHEMA_VERSION_FILE, 'r') as f:
                version_info = json.load(f)
                schema_version = version_info.get('version')

        if searcher:
            # Get document count (approximate)
            doc_count = len(list(DOCS_DIR.glob("**/*.mdx")))

        return {
            "status": "healthy",
            "index_exists": index_exists,
            "schema_version": schema_version,
            "current_schema_version": CURRENT_SCHEMA_VERSION,
            "schema_compatible": schema_version == CURRENT_SCHEMA_VERSION,
            "approximate_documents": doc_count
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# Mount static files for any additional assets
# app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)