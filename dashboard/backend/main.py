from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import uvicorn

from models import (
    DatasetType, DatasetRecord, DatasetRecordSummary, PaginatedResponse,
    DatasetStats, FeedbackSubmission, FeedbackResponse, SearchFilters
)
from database import db_manager

# Initialize database on startup
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    db_manager.load_data()
    yield

app = FastAPI(
    title="Dashboard API", 
    description="API for dashboard datasets", 
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Dashboard API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "total_records": db_manager.get_total_records()}

@app.get("/datasets/{dataset_type}/stats", response_model=DatasetStats)
async def get_dataset_stats(dataset_type: DatasetType):
    """Get statistics for a specific dataset"""
    return db_manager.get_dataset_stats(dataset_type)

@app.get("/datasets/{dataset_type}", response_model=PaginatedResponse)
async def get_datasets(
    dataset_type: DatasetType,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=1000, description="Number of records per page"),
    search_query: Optional[str] = Query(None, description="Search in description and taxonomies"),
    ai_taxonomy: Optional[str] = Query(None, description="Filter by AI taxonomy"),
    current_erms_taxonomy: Optional[str] = Query(None, description="Filter by ERMS taxonomy")
):
    """Get paginated dataset records with search and filtering"""
    result = db_manager.search_records(
        dataset_type=dataset_type,
        search_query=search_query,
        ai_taxonomy=ai_taxonomy,
        current_erms_taxonomy=current_erms_taxonomy,
        page=page,
        page_size=page_size
    )
    return PaginatedResponse(**result)

@app.get("/datasets/{dataset_type}/{record_id}", response_model=DatasetRecord)
async def get_record_detail(dataset_type: DatasetType, record_id: int):
    """Get detailed record information including JSON fields"""
    record = db_manager.get_record_by_id(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    # Verify the record belongs to the requested dataset type
    if record.dataset_type != dataset_type:
        raise HTTPException(status_code=404, detail="Record not found in specified dataset")
    
    return record

@app.get("/datasets/{dataset_type}/taxonomies")
async def get_taxonomies(dataset_type: DatasetType):
    """Get unique taxonomy values for filtering"""
    return db_manager.get_unique_taxonomies(dataset_type)

@app.get("/taxonomies")
async def get_all_taxonomies():
    """Get all unique taxonomy values across datasets"""
    return db_manager.get_unique_taxonomies()

@app.post("/feedback/{record_id}", response_model=FeedbackResponse)
async def submit_feedback(record_id: int, feedback: FeedbackSubmission):
    """Submit feedback for a specific record"""
    # Verify record exists
    record = db_manager.get_record_by_id(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    # Set the record_id in the feedback (override any provided value)
    feedback.record_id = record_id
    
    return db_manager.submit_feedback(feedback)

@app.get("/feedback/{record_id}", response_model=List[FeedbackResponse])
async def get_feedback(record_id: int):
    """Get all feedback for a specific record"""
    # Verify record exists
    record = db_manager.get_record_by_id(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    return db_manager.get_feedback_for_record(record_id)

@app.get("/search", response_model=PaginatedResponse)
async def search_all_datasets(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=1000, description="Number of records per page"),
    search_query: Optional[str] = Query(None, description="Search in description and taxonomies"),
    dataset_type: Optional[DatasetType] = Query(None, description="Filter by dataset type"),
    ai_taxonomy: Optional[str] = Query(None, description="Filter by AI taxonomy"),
    current_erms_taxonomy: Optional[str] = Query(None, description="Filter by ERMS taxonomy")
):
    """Search across all datasets with filtering"""
    result = db_manager.search_records(
        dataset_type=dataset_type,
        search_query=search_query,
        ai_taxonomy=ai_taxonomy,
        current_erms_taxonomy=current_erms_taxonomy,
        page=page,
        page_size=page_size
    )
    return PaginatedResponse(**result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
