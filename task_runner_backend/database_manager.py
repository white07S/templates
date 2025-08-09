import os
import json
import hashlib
import shutil
from typing import Dict, Any, List, Optional
import duckdb
from tinydb import TinyDB, Query
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, base_dir: str = ".", cache_db_path: str = "cache.db"):
        self.base_dir = Path(base_dir)
        self.cache_db_path = cache_db_path
        self.cache_db = TinyDB(cache_db_path)
        self.duckdb_conn = duckdb.connect(":memory:")
        
        # Initialize cache tables
        self.tasks_table = self.cache_db.table('tasks')
        self.results_table = self.cache_db.table('results')
        
    def generate_event_hash(self, description_of_event: str) -> str:
        """Generate a unique hash for an event based on its description."""
        return hashlib.sha256(description_of_event.encode()).hexdigest()[:16]
    
    def create_event_directories(self, data_case: str, event_hash: str, is_processed: bool = False) -> Dict[str, Path]:
        """
        Create directory structure for an event.
        
        Args:
            data_case: The data case (e.g., 'external_loss')
            event_hash: Unique hash for the event
            is_processed: Whether to create in processed or to_process folder
            
        Returns:
            Dict with paths to created directories
        """
        status_dir = "processed" if is_processed else "to_process"
        base_path = self.base_dir / data_case / status_dir / event_hash
        
        meta_data_path = base_path / "meta_data"
        ai_results_path = base_path / "ai_results" if is_processed else None
        
        # Create directories
        meta_data_path.mkdir(parents=True, exist_ok=True)
        if ai_results_path:
            ai_results_path.mkdir(parents=True, exist_ok=True)
        
        paths = {
            "base": base_path,
            "meta_data": meta_data_path,
        }
        
        if ai_results_path:
            paths["ai_results"] = ai_results_path
            
        return paths
    
    def save_event_metadata(self, paths: Dict[str, Path], row_data: Dict[str, Any]) -> None:
        """Save event metadata to details.json."""
        details_path = paths["meta_data"] / "details.json"
        
        # Filter out AI-generated columns if they exist
        metadata = {k: v for k, v in row_data.items() if not k.startswith('ai_')}
        
        with open(details_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.debug(f"Saved metadata to {details_path}")
    
    def save_ai_result(self, paths: Dict[str, Path], task_name: str, result: Dict[str, Any]) -> None:
        """Save AI processing result."""
        if "ai_results" not in paths:
            logger.error("Cannot save AI result: ai_results path not available")
            return
        
        result_path = paths["ai_results"] / f"{task_name}.json"
        
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.debug(f"Saved AI result to {result_path}")
    
    def move_to_processed(self, data_case: str, event_hash: str) -> bool:
        """
        Move an event from to_process to processed folder.
        
        Args:
            data_case: The data case
            event_hash: Event hash
            
        Returns:
            bool: Success status
        """
        try:
            source_path = self.base_dir / data_case / "to_process" / event_hash
            dest_path = self.base_dir / data_case / "processed" / event_hash
            
            if not source_path.exists():
                logger.error(f"Source path does not exist: {source_path}")
                return False
            
            # Create ai_results directory in destination
            dest_path.mkdir(parents=True, exist_ok=True)
            ai_results_path = dest_path / "ai_results"
            ai_results_path.mkdir(exist_ok=True)
            
            # Copy metadata
            shutil.copytree(source_path / "meta_data", dest_path / "meta_data", dirs_exist_ok=True)
            
            # Remove original to_process folder
            shutil.rmtree(source_path)
            
            logger.info(f"Moved event {event_hash} from to_process to processed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to move event {event_hash}: {str(e)}")
            return False
    
    def check_existing_events(self, data_case: str, events: List[str]) -> Dict[str, str]:
        """
        Check which events are already processed or queued for processing.
        
        Args:
            data_case: The data case
            events: List of DescriptionOfEvent strings
            
        Returns:
            Dict mapping event_hash to status ('processed', 'to_process', or 'new')
        """
        event_status = {}
        
        for event in events:
            event_hash = self.generate_event_hash(event)
            
            processed_path = self.base_dir / data_case / "processed" / event_hash
            to_process_path = self.base_dir / data_case / "to_process" / event_hash
            
            if processed_path.exists():
                event_status[event_hash] = 'processed'
            elif to_process_path.exists():
                event_status[event_hash] = 'to_process'
            else:
                event_status[event_hash] = 'new'
        
        return event_status
    
    def get_pending_tasks(self, data_case: str) -> List[Dict[str, Any]]:
        """Get all pending tasks from to_process folder."""
        pending_tasks = []
        to_process_dir = self.base_dir / data_case / "to_process"
        
        if not to_process_dir.exists():
            return pending_tasks
        
        for event_dir in to_process_dir.iterdir():
            if event_dir.is_dir():
                details_path = event_dir / "meta_data" / "details.json"
                if details_path.exists():
                    with open(details_path, 'r') as f:
                        metadata = json.load(f)
                    
                    pending_tasks.append({
                        'event_hash': event_dir.name,
                        'metadata': metadata,
                        'path': event_dir
                    })
        
        return pending_tasks
    
    def cache_task_status(self, task_id: str, status: str, data_case: str, total_items: int, processed_items: int = 0, original_file_path: str = None) -> None:
        """Cache task status in TinyDB for quick retrieval."""
        Task = Query()
        
        task_data = {
            'task_id': task_id,
            'status': status,
            'data_case': data_case,
            'total_items': total_items,
            'processed_items': processed_items,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        if original_file_path:
            task_data['original_file_path'] = original_file_path
        
        # Update existing or insert new
        if self.tasks_table.search(Task.task_id == task_id):
            task_data['updated_at'] = datetime.now().isoformat()
            self.tasks_table.update(task_data, Task.task_id == task_id)
        else:
            self.tasks_table.insert(task_data)
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status from cache."""
        Task = Query()
        result = self.tasks_table.search(Task.task_id == task_id)
        return result[0] if result else None
    
    def cache_llm_result(self, prompt_hash: str, result: Dict[str, Any]) -> None:
        """Cache LLM result to avoid reprocessing."""
        Result = Query()
        
        result_data = {
            'prompt_hash': prompt_hash,
            'result': result,
            'cached_at': datetime.now().isoformat()
        }
        
        # Check if result already exists
        if not self.results_table.search(Result.prompt_hash == prompt_hash):
            self.results_table.insert(result_data)
    
    def get_cached_llm_result(self, prompt_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached LLM result."""
        Result = Query()
        result = self.results_table.search(Result.prompt_hash == prompt_hash)
        return result[0]['result'] if result else None
    
    def cleanup_cache(self, older_than_days: int = 7) -> None:
        """Clean up old cache entries."""
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        cutoff_str = cutoff_date.isoformat()
        
        Task = Query()
        Result = Query()
        
        # Remove old completed tasks
        self.tasks_table.remove((Task.status == 'completed') & (Task.updated_at < cutoff_str))
        
        # Remove old cached results
        self.results_table.remove(Result.cached_at < cutoff_str)
        
        logger.info(f"Cleaned up cache entries older than {older_than_days} days")
    
    def close(self) -> None:
        """Close database connections."""
        self.cache_db.close()
        self.duckdb_conn.close()