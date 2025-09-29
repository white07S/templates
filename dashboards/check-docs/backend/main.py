from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import json
import frontmatter
from typing import Dict, List, Optional
from pydantic import BaseModel

app = FastAPI(title="Documentation API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directory for documentation files
DOCS_DIR = Path(__file__).parent.parent / "src" / "docs"

class DocMetadata(BaseModel):
    id: str
    title: str
    description: str
    category: str
    path: str
    sidebar_position: Optional[int] = None

class DocContent(BaseModel):
    id: str
    title: str
    description: str
    category: str
    content: str
    metadata: Dict

# Documentation metadata
DOCS_METADATA = [
    {
        "id": "getting-started",
        "title": "Getting Started",
        "path": "/docs/getting-started",
        "description": "Quick start guide to get you up and running",
        "category": "Introduction",
        "sidebar_position": 1
    },
    {
        "id": "core-concepts",
        "title": "Core Concepts",
        "path": "/docs/core-concepts",
        "description": "Fundamental concepts and architecture",
        "category": "Fundamentals",
        "sidebar_position": 2
    },
    {
        "id": "api-reference",
        "title": "API Reference",
        "path": "/docs/api-reference",
        "description": "Complete API documentation and endpoints",
        "category": "Reference",
        "sidebar_position": 3
    },
    {
        "id": "examples",
        "title": "Examples",
        "path": "/docs/examples",
        "description": "Practical examples and code snippets",
        "category": "Guides",
        "sidebar_position": 4
    },
    {
        "id": "math-formulas",
        "title": "Math Formulas & Code",
        "path": "/docs/math-formulas",
        "description": "Mathematical formulas and enhanced code blocks",
        "category": "Advanced",
        "sidebar_position": 5
    },
    {
        "id": "troubleshooting",
        "title": "Troubleshooting",
        "path": "/docs/troubleshooting",
        "description": "Common issues and their solutions",
        "category": "Support",
        "sidebar_position": 6
    }
]

@app.get("/")
async def root():
    return {"message": "Documentation API", "version": "1.0.0"}

@app.get("/api/docs", response_model=List[DocMetadata])
async def get_docs_list():
    """Get list of all available documentation"""
    return DOCS_METADATA

@app.get("/api/docs/{doc_id}")
async def get_doc_content(doc_id: str):
    """Get content of a specific documentation file"""

    # Find the doc metadata
    doc_meta = next((doc for doc in DOCS_METADATA if doc["id"] == doc_id), None)
    if not doc_meta:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    # Read the MDX file
    doc_path = DOCS_DIR / f"{doc_id}.mdx"

    if not doc_path.exists():
        raise HTTPException(status_code=404, detail=f"Document file '{doc_id}.mdx' not found")

    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            # Parse frontmatter and content
            post = frontmatter.load(f)

            return {
                "id": doc_id,
                "title": doc_meta["title"],
                "description": doc_meta["description"],
                "category": doc_meta["category"],
                "content": post.content,
                "metadata": post.metadata
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading document: {str(e)}")

@app.get("/api/docs/{doc_id}/raw")
async def get_doc_raw(doc_id: str):
    """Get raw MDX file content"""
    doc_path = DOCS_DIR / f"{doc_id}.mdx"

    if not doc_path.exists():
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}.mdx' not found")

    return FileResponse(doc_path, media_type="text/plain")

@app.get("/api/search")
async def search_docs(q: str):
    """Search documentation content"""
    if not q or len(q) < 2:
        return []

    results = []
    search_term = q.lower()

    for doc_meta in DOCS_METADATA:
        doc_path = DOCS_DIR / f"{doc_meta['id']}.mdx"

        if doc_path.exists():
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                    # Search in title, description, and content
                    if (search_term in doc_meta['title'].lower() or
                        search_term in doc_meta['description'].lower() or
                        search_term in content.lower()):

                        # Extract a snippet around the search term
                        content_lower = content.lower()
                        index = content_lower.find(search_term)
                        snippet = ""

                        if index != -1:
                            start = max(0, index - 100)
                            end = min(len(content), index + 100 + len(search_term))
                            snippet = content[start:end].strip()
                            if start > 0:
                                snippet = "..." + snippet
                            if end < len(content):
                                snippet = snippet + "..."

                        results.append({
                            "id": doc_meta['id'],
                            "title": doc_meta['title'],
                            "description": doc_meta['description'],
                            "category": doc_meta['category'],
                            "path": doc_meta['path'],
                            "snippet": snippet
                        })
            except Exception as e:
                print(f"Error searching {doc_meta['id']}: {e}")
                continue

    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)