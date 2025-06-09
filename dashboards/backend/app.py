from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import duckdb
import pandas as pd
import json
import io
from datetime import datetime, date
from enum import Enum
import uvicorn

app = FastAPI(title="External Loss Data Dashboard API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
conn = duckdb.connect(':memory:')

# Pydantic models
class FilterParams(BaseModel):
    search: Optional[str] = None
    loss_amount_min: Optional[float] = None
    loss_amount_max: Optional[float] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    business_lines: Optional[List[str]] = None
    regions: Optional[List[str]] = None
    countries: Optional[List[str]] = None
    risk_categories: Optional[List[str]] = None
    nfr_taxonomies: Optional[List[str]] = None
    ubs_divisions: Optional[List[str]] = None

class SortParams(BaseModel):
    column: str = "date_of_entry"
    direction: str = "desc"  # asc or desc

class PaginationParams(BaseModel):
    page: int = 1
    page_size: int = 25

class DataResponse(BaseModel):
    data: List[Dict[str, Any]]
    total_count: int
    page: int
    page_size: int
    total_pages: int

class SummaryStats(BaseModel):
    total_losses: float
    total_events: int
    average_loss: float
    affected_firms: int
    highest_loss: float
    recent_events: int

# Initialize database and load data
@app.on_event("startup")
async def startup_event():
    try:
        # Create mock data if not exists
        create_mock_data()
        
        # Load data into DuckDB
        df = pd.read_csv("external_loss_mock_data.csv")
        conn.execute("CREATE TABLE IF NOT EXISTS external_losses AS SELECT * FROM df")
        
        # Create indexes for performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_loss_amount ON external_losses(loss_amount___m_)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_date_entry ON external_losses(date_of_entry)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_region ON external_losses(event_region)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_business_line ON external_losses(basel_business_line__level1_)")
        
        print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")

def create_mock_data():
    """Create mock data if it doesn't exist"""
    import os
    if not os.path.exists("external_loss_mock_data.csv"):
        # Generate mock data using the provided script
        import numpy as np
        from faker import Faker
        import random
        from datetime import datetime, timedelta
        
        fake = Faker()
        taxonomies = ["Fraud", "Cybersecurity", "Data Privacy", "Third-Party Risk", "Process Failures"]
        business_lines = ["Corporate Finance", "Retail Banking", "Wealth Management", "Asset Management"]
        ubs_divisions = ["UBS Europe SE", "UBS AG", "UBS Switzerland", "UBS UK Branch"]
        event_categories = ["External Fraud", "Internal Fraud", "Execution Errors", "Compliance Breach"]
        regions = ["EMEA", "APAC", "AMERICAS"]
        countries = ["Germany", "India", "USA", "UK", "Singapore", "Switzerland"]
        activities = ["Trade Booking", "Client Onboarding", "Transaction Settlement", "Risk Assessment"]
        themes = ["Model Risk", "Automation Risk", "Human Error", "Outsourcing Risk", "Third-Party AI Risk"]
        steps = ["Data Collection", "Preprocessing", "Model Selection", "Evaluation", "Deployment"]
        
        def generate_mock_row():
            settlement_date = fake.date_between(start_date='-3y', end_date='today')
            return {
                "reference_id_code": fake.uuid4(),
                "parent_name": fake.company(),
                "ama_peer_bank": fake.company(),
                "description_of_event": f"{random.choice(taxonomies)} incident occurred in {random.choice(activities)} process.",
                "nfr_taxonomy": random.choice(taxonomies),
                "loss_amount___m_": round(random.uniform(0.1, 100.0), 2),
                "basel_business_line__level1_": random.choice(business_lines),
                "basel_business_line__level_2": random.choice(business_lines),
                "business_unit": fake.bs(),
                "ubs_business_division": random.choice(ubs_divisions),
                "event_risk_category": random.choice(event_categories),
                "sub_risk_category": random.choice(event_categories),
                "activity": random.choice(activities),
                "country_of_incident": random.choice(countries),
                "event_region": random.choice(regions),
                "month__year_of_settlement": settlement_date.strftime("%m/%Y"),
                "multiple_firms_impacted_code": random.choice(["Y", "N"]),
                "single_event_multiple_loss_code": random.choice(["Y", "N"]),
                "date_of_entry": (settlement_date + timedelta(days=random.randint(1, 60))).strftime("%Y-%m-%d"),
                "nfr_taxonomy_number": f"NFR-{random.randint(1000,9999)}",
                "root_cause": random.choice(["Process gap", "System failure", "Insider threat", "Vendor failure"]),
                "learning_outcome": f"The incident highlighted gaps in {random.choice(activities)}.",
                "impact": f"{random.choice(['Severe', 'Moderate', 'Minimal'])} impact on operations and reputational risk.",
                "summary": f"{random.choice(taxonomies)} event causing ${random.randint(1,100)}M loss.",
                "ai_nfr_cluster": random.choice(["AI Ops", "AI Governance", "AI Compliance"]),
                "ai_nfr_taxonomy": random.choice(taxonomies),
                "ai_risk_theme": random.choice(themes),
                "ai_reasoning_taxonomy_steps": ", ".join(random.sample(steps, k=3)),
                "ai_reasoning_risk_theme": f"Risk due to {random.choice(themes)} in {random.choice(activities)}."
            }
        
        data = [generate_mock_row() for _ in range(5000)]
        df = pd.DataFrame(data)
        df.to_csv("external_loss_mock_data.csv", index=False)

def build_where_clause(filters: FilterParams) -> tuple[str, list]:
    """Build WHERE clause and parameters for filtering"""
    conditions = []
    params = []
    
    if filters.search:
        search_fields = [
            "description_of_event", "summary", "parent_name", 
            "learning_outcome", "impact", "root_cause"
        ]
        search_conditions = []
        for field in search_fields:
            search_conditions.append(f"LOWER({field}) LIKE ?")
            params.append(f"%{filters.search.lower()}%")
        conditions.append(f"({' OR '.join(search_conditions)})")
    
    if filters.loss_amount_min is not None:
        conditions.append("loss_amount___m_ >= ?")
        params.append(filters.loss_amount_min)
    
    if filters.loss_amount_max is not None:
        conditions.append("loss_amount___m_ <= ?")
        params.append(filters.loss_amount_max)
    
    if filters.date_from:
        conditions.append("date_of_entry >= ?")
        params.append(filters.date_from)
    
    if filters.date_to:
        conditions.append("date_of_entry <= ?")
        params.append(filters.date_to)
    
    if filters.business_lines:
        placeholders = ','.join(['?' for _ in filters.business_lines])
        conditions.append(f"basel_business_line__level1_ IN ({placeholders})")
        params.extend(filters.business_lines)
    
    if filters.regions:
        placeholders = ','.join(['?' for _ in filters.regions])
        conditions.append(f"event_region IN ({placeholders})")
        params.extend(filters.regions)
    
    if filters.countries:
        placeholders = ','.join(['?' for _ in filters.countries])
        conditions.append(f"country_of_incident IN ({placeholders})")
        params.extend(filters.countries)
    
    if filters.risk_categories:
        placeholders = ','.join(['?' for _ in filters.risk_categories])
        conditions.append(f"event_risk_category IN ({placeholders})")
        params.extend(filters.risk_categories)
    
    if filters.nfr_taxonomies:
        placeholders = ','.join(['?' for _ in filters.nfr_taxonomies])
        conditions.append(f"nfr_taxonomy IN ({placeholders})")
        params.extend(filters.nfr_taxonomies)
    
    if filters.ubs_divisions:
        placeholders = ','.join(['?' for _ in filters.ubs_divisions])
        conditions.append(f"ubs_business_division IN ({placeholders})")
        params.extend(filters.ubs_divisions)
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    return where_clause, params

# API Endpoints

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/summary", response_model=SummaryStats)
async def get_summary_stats(filters: FilterParams = Depends()):
    """Get summary statistics for the dashboard"""
    where_clause, params = build_where_clause(filters)
    
    query = f"""
    SELECT 
        ROUND(SUM(loss_amount___m_), 2) as total_losses,
        COUNT(*) as total_events,
        ROUND(AVG(loss_amount___m_), 2) as average_loss,
        COUNT(DISTINCT parent_name) as affected_firms,
        MAX(loss_amount___m_) as highest_loss,
        COUNT(CASE WHEN date_of_entry >= CURRENT_DATE - INTERVAL '30 days' THEN 1 END) as recent_events
    FROM external_losses 
    WHERE {where_clause}
    """
    
    result = conn.execute(query, params).fetchone()
    
    return SummaryStats(
        total_losses=result[0] or 0,
        total_events=result[1] or 0,
        average_loss=result[2] or 0,
        affected_firms=result[3] or 0,
        highest_loss=result[4] or 0,
        recent_events=result[5] or 0
    )

@app.post("/api/data", response_model=DataResponse)
async def get_data(
    filters: FilterParams = FilterParams(),
    sort: SortParams = SortParams(),
    pagination: PaginationParams = PaginationParams()
):
    """Get paginated and filtered data"""
    where_clause, params = build_where_clause(filters)
    
    # Get total count
    count_query = f"SELECT COUNT(*) FROM external_losses WHERE {where_clause}"
    total_count = conn.execute(count_query, params).fetchone()[0]
    
    # Get paginated data
    offset = (pagination.page - 1) * pagination.page_size
    data_query = f"""
    SELECT * FROM external_losses 
    WHERE {where_clause}
    ORDER BY {sort.column} {sort.direction.upper()}
    LIMIT {pagination.page_size} OFFSET {offset}
    """
    
    results = conn.execute(data_query, params).fetchall()
    columns = [desc[0] for desc in conn.description]
    
    data = [dict(zip(columns, row)) for row in results]
    
    total_pages = (total_count + pagination.page_size - 1) // pagination.page_size
    
    return DataResponse(
        data=data,
        total_count=total_count,
        page=pagination.page,
        page_size=pagination.page_size,
        total_pages=total_pages
    )

@app.get("/api/filter-options")
async def get_filter_options():
    """Get available options for filters"""
    options = {}
    
    # Get unique values for dropdown filters
    queries = {
        "business_lines": "SELECT DISTINCT basel_business_line__level1_ FROM external_losses ORDER BY basel_business_line__level1_",
        "regions": "SELECT DISTINCT event_region FROM external_losses ORDER BY event_region",
        "countries": "SELECT DISTINCT country_of_incident FROM external_losses ORDER BY country_of_incident",
        "risk_categories": "SELECT DISTINCT event_risk_category FROM external_losses ORDER BY event_risk_category",
        "nfr_taxonomies": "SELECT DISTINCT nfr_taxonomy FROM external_losses ORDER BY nfr_taxonomy",
        "ubs_divisions": "SELECT DISTINCT ubs_business_division FROM external_losses ORDER BY ubs_business_division"
    }
    
    for key, query in queries.items():
        results = conn.execute(query).fetchall()
        options[key] = [row[0] for row in results if row[0]]
    
    # Get loss amount range
    loss_range = conn.execute("SELECT MIN(loss_amount___m_), MAX(loss_amount___m_) FROM external_losses").fetchone()
    options["loss_amount_range"] = {"min": loss_range[0], "max": loss_range[1]}
    
    # Get date range
    date_range = conn.execute("SELECT MIN(date_of_entry), MAX(date_of_entry) FROM external_losses").fetchone()
    options["date_range"] = {"min": date_range[0], "max": date_range[1]}
    
    return options

@app.get("/api/charts/time-series")
async def get_time_series_data(filters: FilterParams = Depends()):
    """Get time series data for loss trends"""
    where_clause, params = build_where_clause(filters)
    
    query = f"""
    SELECT 
        DATE_TRUNC('month', CAST(date_of_entry AS DATE)) as month,
        COUNT(*) as event_count,
        ROUND(SUM(loss_amount___m_), 2) as total_loss
    FROM external_losses 
    WHERE {where_clause}
    GROUP BY DATE_TRUNC('month', CAST(date_of_entry AS DATE))
    ORDER BY month
    """
    
    results = conn.execute(query, params).fetchall()
    
    return [
        {
            "month": row[0].strftime("%Y-%m") if row[0] else "",
            "event_count": row[1],
            "total_loss": row[2]
        }
        for row in results
    ]

@app.get("/api/charts/distribution")
async def get_distribution_data(
    dimension: str = Query(..., description="Dimension to group by"),
    filters: FilterParams = Depends()
):
    """Get distribution data for various dimensions"""
    where_clause, params = build_where_clause(filters)
    
    # Map dimension names to actual column names
    dimension_map = {
        "business_line": "basel_business_line__level1_",
        "region": "event_region",
        "risk_category": "event_risk_category",
        "nfr_taxonomy": "nfr_taxonomy",
        "country": "country_of_incident"
    }
    
    column = dimension_map.get(dimension)
    if not column:
        raise HTTPException(status_code=400, detail="Invalid dimension")
    
    query = f"""
    SELECT 
        {column} as category,
        COUNT(*) as event_count,
        ROUND(SUM(loss_amount___m_), 2) as total_loss,
        ROUND(AVG(loss_amount___m_), 2) as avg_loss
    FROM external_losses 
    WHERE {where_clause}
    GROUP BY {column}
    ORDER BY total_loss DESC
    LIMIT 20
    """
    
    results = conn.execute(query, params).fetchall()
    
    return [
        {
            "category": row[0],
            "event_count": row[1],
            "total_loss": row[2],
            "avg_loss": row[3]
        }
        for row in results
    ]

@app.get("/api/charts/top-losses")
async def get_top_losses(
    limit: int = Query(10, ge=1, le=50),
    filters: FilterParams = Depends()
):
    """Get top loss events"""
    where_clause, params = build_where_clause(filters)
    
    query = f"""
    SELECT 
        reference_id_code,
        parent_name,
        loss_amount___m_,
        nfr_taxonomy,
        event_region,
        date_of_entry,
        description_of_event
    FROM external_losses 
    WHERE {where_clause}
    ORDER BY loss_amount___m_ DESC
    LIMIT {limit}
    """
    
    results = conn.execute(query, params).fetchall()
    columns = [desc[0] for desc in conn.description]
    
    return [dict(zip(columns, row)) for row in results]

@app.get("/api/export/csv")
async def export_csv(filters: FilterParams = Depends()):
    """Export filtered data as CSV"""
    where_clause, params = build_where_clause(filters)
    
    query = f"SELECT * FROM external_losses WHERE {where_clause}"
    results = conn.execute(query, params).fetchall()
    columns = [desc[0] for desc in conn.description]
    
    # Create CSV
    df = pd.DataFrame(results, columns=columns)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    return StreamingResponse(
        io.BytesIO(csv_content.encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=external_losses_export.csv"}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)