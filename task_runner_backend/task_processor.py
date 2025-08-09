import json
import uuid
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from datetime import datetime
import threading
from tqdm import tqdm
import os

from file_validator import FileValidator
from database_manager import DatabaseManager
from llm_processor import LLMProcessor

logger = logging.getLogger(__name__)

class TaskProcessor:
    def __init__(self, config_path: str = "config.json", max_workers: int = None, 
                 use_multiprocessing: bool = True):
        """
        Initialize the task processor.
        
        Args:
            config_path: Path to config file
            max_workers: Maximum number of worker threads/processes
            use_multiprocessing: Whether to use multiprocessing vs threading
        """
        self.validator = FileValidator(config_path)
        self.db_manager = DatabaseManager()
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.use_multiprocessing = use_multiprocessing
        
        # Thread-safe progress tracking
        self._progress_lock = threading.Lock()
        self._progress_data = {}
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def generate_task_id(self) -> str:
        """Generate unique task ID."""
        return str(uuid.uuid4())
    
    def prepare_data_for_processing(self, df: pd.DataFrame, task: str, data_case: str, 
                                   task_id: str, include_to_process: bool = False) -> List[Dict[str, Any]]:
        """
        Prepare data for processing by checking existing events and creating directory structure.
        
        Args:
            df: Input dataframe
            task: Task name
            data_case: Data case name
            task_id: Unique task ID
            include_to_process: Whether to include events already in to_process folder
            
        Returns:
            List of processing jobs
        """
        # Get list of events and check existing status
        events = df['DescriptionOfEvent'].tolist()
        event_status = self.db_manager.check_existing_events(data_case, events)
        
        # Create processing jobs for new events and optionally to_process events
        processing_jobs = []
        
        for idx, row in df.iterrows():
            event_hash = self.db_manager.generate_event_hash(row['DescriptionOfEvent'])
            status = event_status.get(event_hash, 'new')
            
            if status == 'new' or (include_to_process and status == 'to_process'):
                if status == 'new':
                    # Create directory structure for new event
                    paths = self.db_manager.create_event_directories(data_case, event_hash, is_processed=False)
                    # Save metadata
                    self.db_manager.save_event_metadata(paths, row.to_dict())
                else:
                    # Use existing to_process directory
                    paths = {
                        "base": self.db_manager.base_dir / data_case / "to_process" / event_hash,
                        "meta_data": self.db_manager.base_dir / data_case / "to_process" / event_hash / "meta_data"
                    }
                
                # Add to processing queue
                processing_jobs.append({
                    'event_hash': event_hash,
                    'row_data': row.to_dict(),
                    'paths': paths,
                    'task': task,
                    'data_case': data_case,
                    'task_id': task_id
                })
        
        new_events = sum(1 for _, row in df.iterrows() if event_status.get(self.db_manager.generate_event_hash(row['DescriptionOfEvent']), 'new') == 'new')
        to_process_events = len(processing_jobs) - new_events if include_to_process else 0
        
        logger.info(f"Prepared {len(processing_jobs)} events for processing ({new_events} new, {to_process_events} from to_process)")
        logger.info(f"Skipped {len(events) - len(processing_jobs)} already processed events")
        
        return processing_jobs
    
    def process_single_job(self, job: Dict[str, Any], llm_processor: LLMProcessor) -> Dict[str, Any]:
        """
        Process a single job (event).
        
        Args:
            job: Processing job data
            llm_processor: LLM processor instance
            
        Returns:
            Processing result
        """
        try:
            event_hash = job['event_hash']
            row_data = job['row_data']
            task = job['task']
            data_case = job['data_case']
            task_id = job['task_id']
            
            # Get task configuration
            task_config = self.config['tasks'][task]
            prompt_file = task_config['prompt']
            
            # Process with LLM
            result = llm_processor.process_single_row(
                row_data, task, data_case, prompt_file, self.db_manager
            )
            
            # Move to processed and save AI results
            processed_paths = self.db_manager.create_event_directories(
                data_case, event_hash, is_processed=True
            )
            
            # Copy metadata
            self.db_manager.save_event_metadata(processed_paths, row_data)
            
            # Save AI result
            self.db_manager.save_ai_result(processed_paths, task, result['result'])
            
            # Remove from to_process
            self.db_manager.move_to_processed(data_case, event_hash)
            
            # Update progress
            with self._progress_lock:
                if task_id not in self._progress_data:
                    self._progress_data[task_id] = {'completed': 0, 'errors': 0}
                self._progress_data[task_id]['completed'] += 1
            
            return {
                'event_hash': event_hash,
                'status': 'success',
                'result': result,
                'processing_time': result.get('processing_time', 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to process job {job.get('event_hash', 'unknown')}: {str(e)}")
            
            # Update error count
            with self._progress_lock:
                if task_id not in self._progress_data:
                    self._progress_data[task_id] = {'completed': 0, 'errors': 0}
                self._progress_data[task_id]['errors'] += 1
            
            return {
                'event_hash': job.get('event_hash', 'unknown'),
                'status': 'error',
                'error': str(e)
            }
    
    def process_with_threading(self, jobs: List[Dict[str, Any]], llm_processor: LLMProcessor, 
                             progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Process jobs using ThreadPoolExecutor."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self.process_single_job, job, llm_processor): job 
                for job in jobs
            }
            
            # Process completed jobs
            with tqdm(total=len(jobs), desc="Processing events") as pbar:
                for future in as_completed(future_to_job):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    
                    if progress_callback:
                        progress_callback(len(results), len(jobs), result)
        
        return results
    
    def process_with_multiprocessing(self, jobs: List[Dict[str, Any]], llm_processor: LLMProcessor,
                                   progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Process jobs using ProcessPoolExecutor."""
        results = []
        
        # Note: For multiprocessing, we need to pass serializable data
        # The LLMProcessor needs to be recreated in each process
        serializable_jobs = []
        for job in jobs:
            serializable_job = job.copy()
            # Pass the config path so LLMProcessor can be recreated with same config
            serializable_job['config_path'] = 'config.json'
            serializable_jobs.append(serializable_job)
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            futures = [
                executor.submit(self._process_job_multiprocessing, job) 
                for job in serializable_jobs
            ]
            
            # Process completed jobs
            with tqdm(total=len(jobs), desc="Processing events") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    
                    if progress_callback:
                        progress_callback(len(results), len(jobs), result)
        
        return results
    
    @staticmethod
    def _process_job_multiprocessing(job: Dict[str, Any]) -> Dict[str, Any]:
        """Static method for multiprocessing job processing."""
        # This method runs in a separate process
        # Recreate necessary instances
        try:
            from database_manager import DatabaseManager
            from llm_processor import LLMProcessor
            
            config_path = job.pop('config_path', 'config.json')
            llm_processor = LLMProcessor(config_path=config_path)
            
            db_manager = DatabaseManager()
            
            # Create a temporary task processor for this job
            processor = TaskProcessor()
            return processor.process_single_job(job, llm_processor)
            
        except Exception as e:
            return {
                'event_hash': job.get('event_hash', 'unknown'),
                'status': 'error',
                'error': str(e)
            }
    
    def resume_pending_tasks(self, task: str, data_case: str, llm_processor: LLMProcessor,
                           progress_callback: Optional[Callable] = None) -> str:
        """
        Resume processing of tasks stuck in to_process folder.
        
        Args:
            task: Task name
            data_case: Data case name  
            llm_processor: LLM processor instance
            progress_callback: Optional callback for progress updates
            
        Returns:
            str: Task ID
        """
        task_id = self.generate_task_id()
        
        try:
            # Get all pending tasks from to_process folder
            pending_tasks = self.db_manager.get_pending_tasks(data_case)
            
            if not pending_tasks:
                logger.info("No pending tasks found in to_process folder")
                self.db_manager.cache_task_status(task_id, "completed", data_case, 0, 0)
                return task_id
            
            # Convert pending tasks to processing jobs format
            jobs = []
            for pending_task in pending_tasks:
                event_hash = pending_task['event_hash']
                row_data = pending_task['metadata']
                
                paths = {
                    "base": self.db_manager.base_dir / data_case / "to_process" / event_hash,
                    "meta_data": self.db_manager.base_dir / data_case / "to_process" / event_hash / "meta_data"
                }
                
                jobs.append({
                    'event_hash': event_hash,
                    'row_data': row_data,
                    'paths': paths,
                    'task': task,
                    'data_case': data_case,
                    'task_id': task_id
                })
            
            logger.info(f"Found {len(jobs)} pending tasks to resume processing")
            
            # Cache initial task status
            self.db_manager.cache_task_status(task_id, "processing", data_case, len(jobs), 0)
            
            # Process jobs
            if self.use_multiprocessing and len(jobs) > 5:
                results = self.process_with_multiprocessing(jobs, llm_processor, progress_callback)
            else:
                results = self.process_with_threading(jobs, llm_processor, progress_callback)
            
            # Update final status
            successful = sum(1 for r in results if r['status'] == 'success')
            failed = len(results) - successful
            
            final_status = "completed" if failed == 0 else "completed_with_errors"
            # Resume tasks don't have original file path
            self.db_manager.cache_task_status(task_id, final_status, data_case, len(jobs), successful)
            
            logger.info(f"Resume task {task_id} completed: {successful} successful, {failed} failed")
            
            return task_id
            
        except Exception as e:
            logger.error(f"Resume task {task_id} failed: {str(e)}")
            self.db_manager.cache_task_status(task_id, "failed", data_case, 0, 0)
            raise

    def submit_task(self, file_path: str, task: str, data_case: str, 
                   llm_processor: LLMProcessor, progress_callback: Optional[Callable] = None,
                   include_to_process: bool = False) -> str:
        """
        Submit a new task for processing.
        
        Args:
            file_path: Path to uploaded file
            task: Task name
            data_case: Data case name
            llm_processor: LLM processor instance
            progress_callback: Optional callback for progress updates
            
        Returns:
            str: Task ID
        """
        # Generate task ID
        task_id = self.generate_task_id()
        
        try:
            # Validate file and submission
            df, is_valid, missing_columns = self.validator.validate_file_submission(
                file_path, task, data_case
            )
            
            if not is_valid:
                raise ValueError(f"File validation failed. Missing columns: {missing_columns}")
            
            # Prepare processing jobs
            jobs = self.prepare_data_for_processing(df, task, data_case, task_id, include_to_process)
            
            if not jobs:
                logger.info("No new events to process")
                self.db_manager.cache_task_status(task_id, "completed", data_case, 0, 0, file_path)
                return task_id
            
            # Cache initial task status
            self.db_manager.cache_task_status(task_id, "processing", data_case, len(jobs), 0, file_path)
            
            # Process jobs
            if self.use_multiprocessing and len(jobs) > 5:
                results = self.process_with_multiprocessing(jobs, llm_processor, progress_callback)
            else:
                results = self.process_with_threading(jobs, llm_processor, progress_callback)
            
            # Update final status
            successful = sum(1 for r in results if r['status'] == 'success')
            failed = len(results) - successful
            
            final_status = "completed" if failed == 0 else "completed_with_errors"
            self.db_manager.cache_task_status(task_id, final_status, data_case, len(jobs), successful, file_path)
            
            logger.info(f"Task {task_id} completed: {successful} successful, {failed} failed")
            
            return task_id
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {str(e)}")
            self.db_manager.cache_task_status(task_id, "failed", data_case, 0, 0, file_path)
            raise
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get current status of a task."""
        status = self.db_manager.get_task_status(task_id)
        
        if not status:
            return {"error": "Task not found"}
        
        # Add progress percentage
        if status['total_items'] > 0:
            status['progress_percentage'] = (status['processed_items'] / status['total_items']) * 100
        else:
            status['progress_percentage'] = 100
        
        return status
    
    def export_results(self, original_file_path: str, task_id: str, task: str, data_case: str) -> str:
        """
        Export results by adding AI-generated columns to original file.
        
        Args:
            original_file_path: Path to original uploaded file
            task_id: Task ID
            task: Task name
            data_case: Data case name
            
        Returns:
            str: Path to exported file
        """
        # Load original file
        df = pd.read_csv(original_file_path) if original_file_path.endswith('.csv') else pd.read_excel(original_file_path)
        
        # Add AI results columns
        ai_results = []
        
        for idx, row in df.iterrows():
            event_hash = self.db_manager.generate_event_hash(row['DescriptionOfEvent'])
            
            # Look for processed result
            processed_path = self.db_manager.base_dir / data_case / "processed" / event_hash / "ai_results" / f"{task}.json"
            
            if processed_path.exists():
                with open(processed_path, 'r') as f:
                    result = json.load(f)
                    ai_results.append(result)
            else:
                # No result found
                ai_results.append({
                    'summary': 'Not processed',
                    'key_points': [],
                    'risk_level': 'Unknown',
                    'impact_assessment': 'Not processed'
                })
        
        # Add AI columns to dataframe
        for key in ['summary', 'key_points', 'risk_level', 'impact_assessment']:
            df[f'ai_{key}'] = [result.get(key, '') for result in ai_results]
        
        # Convert key_points list to string
        df['ai_key_points'] = df['ai_key_points'].apply(lambda x: '; '.join(x) if isinstance(x, list) else str(x))
        
        # Export to new file
        export_path = f"exports/task_{task_id}_{task}_{data_case}_results.csv"
        os.makedirs("exports", exist_ok=True)
        df.to_csv(export_path, index=False)
        
        logger.info(f"Results exported to {export_path}")
        return export_path