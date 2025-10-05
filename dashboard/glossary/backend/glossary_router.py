from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
from tinydb import Query as TinyQuery
import json
from models import Term, TermCreate, TermUpdate, TermResponse
from database import db, temp_db, get_next_id, initialize_database

router = APIRouter(prefix="/api/glossary", tags=["glossary"])

# Initialize database on startup
initialize_database()

@router.post("/terms", response_model=TermResponse)
async def create_term(term_data: TermCreate):
    """Create a new term"""
    # Check for duplicate term
    Term_query = TinyQuery()
    existing = db.search(Term_query.term.matches(term_data.term, flags=0))

    if existing:
        raise HTTPException(status_code=400, detail="Term already exists")

    # Create new term
    new_term = {
        "id": get_next_id(),
        "term": term_data.term,
        "definition": term_data.definition,
        "synonyms": term_data.synonyms if term_data.synonyms else [],
        "createdAt": datetime.now().isoformat(),
        "updatedAt": datetime.now().isoformat()
    }

    # Add to temp database for review
    temp_entry = {
        **new_term,
        "action": "create",
        "user_id": term_data.user_id,
        "timestamp": datetime.now().isoformat()
    }
    temp_db.insert(temp_entry)

    # Also add to main database (for demo purposes)
    db.insert(new_term)

    return TermResponse(**new_term)

@router.get("/terms", response_model=dict)
async def get_all_terms(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Items per page")
):
    """View all terms with pagination"""
    all_terms = db.all()
    # Sort by term alphabetically
    all_terms.sort(key=lambda x: x["term"].lower())

    # Calculate pagination
    total_items = len(all_terms)
    total_pages = (total_items + limit - 1) // limit  # Ceiling division
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit

    # Get paginated terms
    paginated_terms = all_terms[start_idx:end_idx]

    return {
        "items": [TermResponse(**term) for term in paginated_terms],
        "page": page,
        "limit": limit,
        "total_items": total_items,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1
    }

@router.get("/terms/{term_id}", response_model=TermResponse)
async def get_term(term_id: int):
    """View a single term by ID"""
    Term_query = TinyQuery()
    term = db.search(Term_query.id == term_id)

    if not term:
        raise HTTPException(status_code=404, detail="Term not found")

    return TermResponse(**term[0])

@router.put("/terms/{term_id}", response_model=TermResponse)
async def update_term(term_id: int, term_data: TermUpdate):
    """Update a term by ID"""
    Term_query = TinyQuery()
    existing = db.search(Term_query.id == term_id)

    if not existing:
        raise HTTPException(status_code=404, detail="Term not found")

    # Check for duplicate term name if term is being changed
    if term_data.term and term_data.term != existing[0]["term"]:
        duplicate = db.search(Term_query.term.matches(term_data.term, flags=0))
        if duplicate:
            raise HTTPException(status_code=400, detail="Term name already exists")

    # Prepare update data
    updated_term = existing[0].copy()
    if term_data.term is not None:
        updated_term["term"] = term_data.term
    if term_data.definition is not None:
        updated_term["definition"] = term_data.definition
    if term_data.synonyms is not None:
        updated_term["synonyms"] = term_data.synonyms
    updated_term["updatedAt"] = datetime.now().isoformat()

    # Add to temp database for review
    temp_entry = {
        **updated_term,
        "action": "update",
        "user_id": term_data.user_id,
        "original": existing[0],
        "timestamp": datetime.now().isoformat()
    }
    temp_db.insert(temp_entry)

    # Update in main database (for demo purposes)
    db.update(updated_term, Term_query.id == term_id)

    return TermResponse(**updated_term)

@router.get("/search", response_model=dict)
async def search_terms(
    q: str = Query(..., description="Search query"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Items per page")
):
    """Search terms by term name (partial match) with pagination"""
    if not q:
        return {
            "items": [],
            "page": page,
            "limit": limit,
            "total_items": 0,
            "total_pages": 0,
            "has_next": False,
            "has_prev": False
        }

    Term_query = TinyQuery()
    # Case-insensitive partial match in both term and synonyms
    results = db.search(
        (Term_query.term.test(lambda v: q.lower() in v.lower())) |
        (Term_query.synonyms.test(lambda synonyms: any(q.lower() in syn.lower() for syn in (synonyms or []))))
    )

    # Sort results by relevance (exact match first, then alphabetical)
    results.sort(key=lambda x: (
        not x["term"].lower().startswith(q.lower()),
        x["term"].lower()
    ))

    # Calculate pagination
    total_items = len(results)
    total_pages = (total_items + limit - 1) // limit if total_items > 0 else 0
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit

    # Get paginated results
    paginated_results = results[start_idx:end_idx]

    return {
        "items": [TermResponse(**term) for term in paginated_results],
        "page": page,
        "limit": limit,
        "total_items": total_items,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1
    }

@router.delete("/terms/{term_id}")
async def delete_term(term_id: int):
    """Delete a term by ID"""
    Term_query = TinyQuery()
    existing = db.search(Term_query.id == term_id)

    if not existing:
        raise HTTPException(status_code=404, detail="Term not found")

    # Add deletion record to temp database
    temp_entry = {
        "id": term_id,
        "action": "delete",
        "term": existing[0],
        "timestamp": datetime.now().isoformat()
    }
    temp_db.insert(temp_entry)

    # Remove from main database
    db.remove(Term_query.id == term_id)

    return {"message": "Term deleted successfully", "id": term_id}

@router.get("/pending-changes")
async def get_pending_changes():
    """Get all pending changes from temp database for review"""
    return temp_db.all()