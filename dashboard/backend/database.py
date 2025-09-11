import duckdb
import pandas as pd
from tinydb import TinyDB, Query
from typing import List, Optional, Dict, Any
from models import DatasetType, DatasetRecord, DatasetRecordSummary, FeedbackSubmission, FeedbackResponse
import json
import os
from datetime import datetime
import uuid

class DatabaseManager:
    def __init__(self):
        self.conn = duckdb.connect(':memory:')  # In-memory for fast queries
        self.feedback_base_dir = '/Users/preetam/Develop/dashboard/backend/feedback'
        self.data_loaded = False
        
        # Create feedback directories
        os.makedirs(self.feedback_base_dir, exist_ok=True)
        for dataset_type in ['external_loss', 'internal_loss', 'issues', 'controls']:
            os.makedirs(os.path.join(self.feedback_base_dir, dataset_type), exist_ok=True)
        
    def load_data(self):
        """Load CSV data into DuckDB"""
        if self.data_loaded:
            return
            
        try:
            # Load the combined CSV data
            csv_path = '/Users/preetam/Develop/dashboard/backend/dashboard_data.csv'
            
            # Create table and load data
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS datasets AS 
                SELECT * FROM read_csv_auto(?)
            """, [csv_path])
            
            # Create indexes for performance
            self.conn.execute("CREATE INDEX idx_dataset_type ON datasets(dataset_type)")
            self.conn.execute("CREATE INDEX idx_ai_taxonomy ON datasets(ai_taxonomy)")
            self.conn.execute("CREATE INDEX idx_erms_taxonomy ON datasets(current_erms_taxonomy)")
            
            self.data_loaded = True
            print(f"Loaded data successfully. Total records: {self.get_total_records()}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def get_total_records(self) -> int:
        """Get total number of records"""
        result = self.conn.execute("SELECT COUNT(*) FROM datasets").fetchone()
        return result[0] if result else 0
    
    def get_dataset_stats(self, dataset_type: DatasetType) -> Dict[str, Any]:
        """Get statistics for a specific dataset type"""
        query = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT ai_taxonomy) as unique_ai_taxonomies,
            COUNT(DISTINCT current_erms_taxonomy) as unique_erms_taxonomies
        FROM datasets 
        WHERE dataset_type = ?
        """
        
        result = self.conn.execute(query, [dataset_type.value]).fetchone()
        
        return {
            "dataset_type": dataset_type,
            "total_records": result[0],
            "unique_ai_taxonomies": result[1], 
            "unique_erms_taxonomies": result[2]
        }
    
    def search_records(
        self, 
        dataset_type: Optional[DatasetType] = None,
        search_query: Optional[str] = None,
        ai_taxonomy: Optional[str] = None,
        current_erms_taxonomy: Optional[str] = None,
        page: int = 1,
        page_size: int = 50
    ) -> Dict[str, Any]:
        """Search and paginate records"""
        
        # Build WHERE clause
        conditions = []
        params = []
        
        if dataset_type:
            conditions.append("dataset_type = ?")
            params.append(dataset_type.value)
        
        if search_query:
            conditions.append("(description ILIKE ? OR ai_taxonomy ILIKE ? OR current_erms_taxonomy ILIKE ?)")
            search_pattern = f"%{search_query}%"
            params.extend([search_pattern, search_pattern, search_pattern])
        
        if ai_taxonomy:
            conditions.append("ai_taxonomy = ?")
            params.append(ai_taxonomy)
            
        if current_erms_taxonomy:
            conditions.append("current_erms_taxonomy = ?")
            params.append(current_erms_taxonomy)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        # Count total matching records
        count_query = f"SELECT COUNT(*) FROM datasets {where_clause}"
        total = self.conn.execute(count_query, params).fetchone()[0]
        
        # Get paginated data
        offset = (page - 1) * page_size
        data_query = f"""
        SELECT id, description, ai_taxonomy, current_erms_taxonomy 
        FROM datasets 
        {where_clause}
        ORDER BY id
        LIMIT ? OFFSET ?
        """
        
        results = self.conn.execute(data_query, params + [page_size, offset]).fetchall()
        
        records = [
            DatasetRecordSummary(
                id=row[0],
                description=row[1],
                ai_taxonomy=row[2], 
                current_erms_taxonomy=row[3]
            ) for row in results
        ]
        
        return {
            "data": records,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }
    
    def get_record_by_id(self, record_id: int) -> Optional[DatasetRecord]:
        """Get a single record with full details"""
        query = "SELECT * FROM datasets WHERE id = ?"
        result = self.conn.execute(query, [record_id]).fetchone()
        
        if not result:
            return None
        
        # Parse JSON fields
        try:
            raw_meta_data = json.loads(result[5]) if result[5] else None
        except (json.JSONDecodeError, TypeError):
            raw_meta_data = None
            
        try:
            ai_root_cause = json.loads(result[7]) if result[7] else None
        except (json.JSONDecodeError, TypeError):
            ai_root_cause = None
            
        try:
            ai_enrichment = json.loads(result[8]) if result[8] else None
        except (json.JSONDecodeError, TypeError):
            ai_enrichment = None
        
        return DatasetRecord(
            id=result[0],
            dataset_type=DatasetType(result[1]),
            description=result[2],
            ai_taxonomy=result[3],
            current_erms_taxonomy=result[4],
            raw_meta_data=raw_meta_data,
            ai_root_cause=ai_root_cause,
            ai_enrichment=ai_enrichment
        )
    
    def get_unique_taxonomies(self, dataset_type: Optional[DatasetType] = None) -> Dict[str, List[str]]:
        """Get unique taxonomy values for filtering"""
        where_clause = "WHERE dataset_type = ?" if dataset_type else ""
        params = [dataset_type.value] if dataset_type else []
        
        ai_query = f"SELECT DISTINCT ai_taxonomy FROM datasets {where_clause} ORDER BY ai_taxonomy"
        erms_query = f"SELECT DISTINCT current_erms_taxonomy FROM datasets {where_clause} ORDER BY current_erms_taxonomy"
        
        ai_taxonomies = [row[0] for row in self.conn.execute(ai_query, params).fetchall()]
        erms_taxonomies = [row[0] for row in self.conn.execute(erms_query, params).fetchall()]
        
        return {
            "ai_taxonomies": ai_taxonomies,
            "erms_taxonomies": erms_taxonomies
        }
    
    def _get_feedback_file_path(self, record_id: int) -> str:
        """Get the feedback file path for a record"""
        # First, find which dataset this record belongs to
        query = "SELECT dataset_type FROM datasets WHERE id = ?"
        result = self.conn.execute(query, [record_id]).fetchone()
        
        if not result:
            raise ValueError(f"Record {record_id} not found")
        
        dataset_type = result[0]
        return os.path.join(self.feedback_base_dir, dataset_type, f"{record_id}_feedback.json")
    
    def submit_feedback(self, feedback: FeedbackSubmission) -> FeedbackResponse:
        """Submit feedback to individual JSON file"""
        feedback_id = str(uuid.uuid4())
        
        feedback_record = {
            "id": feedback_id,
            "record_id": feedback.record_id,
            "feedback_type": feedback.feedback_type.value,
            "value": feedback.value,
            "additional_notes": feedback.additional_notes,
            "timestamp": datetime.now().isoformat()
        }
        
        feedback_file = self._get_feedback_file_path(feedback.record_id)
        
        # Load existing feedback or create new list
        existing_feedback = []
        if os.path.exists(feedback_file):
            try:
                with open(feedback_file, 'r') as f:
                    existing_feedback = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing_feedback = []
        
        # Add new feedback
        existing_feedback.append(feedback_record)
        
        # Save back to file
        with open(feedback_file, 'w') as f:
            json.dump(existing_feedback, f, indent=2)
        
        return FeedbackResponse(
            id=feedback_id,
            record_id=feedback.record_id,
            feedback_type=feedback.feedback_type,
            value=feedback.value,
            additional_notes=feedback.additional_notes,
            timestamp=datetime.fromisoformat(feedback_record["timestamp"])
        )
    
    def get_feedback_for_record(self, record_id: int) -> List[FeedbackResponse]:
        """Get all feedback for a specific record from JSON file"""
        try:
            feedback_file = self._get_feedback_file_path(record_id)
            
            if not os.path.exists(feedback_file):
                return []
            
            with open(feedback_file, 'r') as f:
                feedback_data = json.load(f)
            
            return [
                FeedbackResponse(
                    id=item["id"],
                    record_id=item["record_id"],
                    feedback_type=item["feedback_type"],
                    value=item["value"],
                    additional_notes=item.get("additional_notes"),
                    timestamp=datetime.fromisoformat(item["timestamp"])
                ) for item in feedback_data
            ]
            
        except (ValueError, json.JSONDecodeError, IOError) as e:
            print(f"Error loading feedback for record {record_id}: {e}")
            return []

# Global database instance
db_manager = DatabaseManager()