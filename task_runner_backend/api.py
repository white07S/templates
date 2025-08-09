import os
import shutil
from typing import Optional, Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import logging
from datetime import datetime

from task_processor import TaskProcessor
from llm_processor import LLMProcessor
from file_validator import FileValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Task Runner Backend", version="1.0.0")

# Global instances
task_processor = TaskProcessor()
llm_processor = None  # Will be initialized when API key is provided

# Request/Response models
class TaskSubmission(BaseModel):
    task: str
    data_case: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress_percentage: float
    total_items: int
    processed_items: int
    created_at: str
    updated_at: str

class TaskSubmissionResponse(BaseModel):
    task_id: str
    status: str
    message: str

# Background task storage
background_tasks_status = {}

def initialize_llm_processor():
    """Initialize LLM processor with configuration from config.json."""
    try:
        return LLMProcessor()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize LLM processor: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global llm_processor
    try:
        llm_processor = initialize_llm_processor()
        logger.info("LLM processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LLM processor: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Task Runner Backend API",
        "version": "1.0.0",
        "endpoints": {
            "submit_task": "/submit",
            "check_status": "/status/{task_id}",
            "download_results": "/download/{task_id}",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "llm_processor": llm_processor is not None,
            "task_processor": True
        }
    }

async def process_task_background(file_path: str, task: str, data_case: str, task_id: str):
    """Background task for processing uploaded file."""
    try:
        logger.info(f"Starting background processing for task {task_id}")
        
        def progress_callback(completed: int, total: int, result: Dict[str, Any]):
            """Callback to update progress."""
            progress = (completed / total) * 100 if total > 0 else 0
            background_tasks_status[task_id] = {
                'status': 'processing',
                'progress': progress,
                'completed': completed,
                'total': total,
                'last_update': datetime.now().isoformat()
            }
        
        # Process the task
        final_task_id = task_processor.submit_task(
            file_path, task, data_case, llm_processor, progress_callback
        )
        
        # Update final status
        background_tasks_status[task_id] = {
            'status': 'completed',
            'progress': 100,
            'completed': background_tasks_status.get(task_id, {}).get('total', 0),
            'total': background_tasks_status.get(task_id, {}).get('total', 0),
            'last_update': datetime.now().isoformat(),
            'final_task_id': final_task_id
        }
        
        logger.info(f"Background processing completed for task {task_id}")
        
    except Exception as e:
        logger.error(f"Background processing failed for task {task_id}: {str(e)}")
        background_tasks_status[task_id] = {
            'status': 'failed',
            'error': str(e),
            'last_update': datetime.now().isoformat()
        }
    finally:
        # Keep the uploaded file for download - don't delete it here
        logger.info(f"Processing completed for task {task_id}, file retained for download")

@app.post("/submit", response_model=TaskSubmissionResponse)
async def submit_task(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    task: str = Form(...),
    data_case: str = Form(...)
):
    """
    Submit a new task for processing.
    
    Args:
        file: Uploaded CSV/Excel file
        task: Task name (e.g., 'summarize')
        data_case: Data case (e.g., 'external_loss')
        
    Returns:
        Task submission response with task_id
    """
    if not llm_processor:
        raise HTTPException(status_code=500, detail="LLM processor not initialized")
    
    try:
        # Validate file format
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")
        
        # Generate task ID
        task_id = task_processor.generate_task_id()
        
        # Save uploaded file temporarily
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, f"{task_id}_{file.filename}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Validate task and data case
        validator = FileValidator()
        validator.validate_task_and_data_case(task, data_case)
        
        # Initialize background task status
        background_tasks_status[task_id] = {
            'status': 'pending',
            'progress': 0,
            'completed': 0,
            'total': 0,
            'created_at': datetime.now().isoformat()
        }
        
        # Add background task
        background_tasks.add_task(process_task_background, file_path, task, data_case, task_id)
        
        logger.info(f"Task {task_id} submitted successfully")
        
        return TaskSubmissionResponse(
            task_id=task_id,
            status="pending",
            message="Task submitted successfully. Use /status/{task_id} to check progress."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Get current status of a task.
    
    Args:
        task_id: Task ID from submission
        
    Returns:
        Task status information
    """
    try:
        # Check background task status first
        if task_id in background_tasks_status:
            bg_status = background_tasks_status[task_id]
            
            if bg_status['status'] == 'completed' and 'final_task_id' in bg_status:
                # Get detailed status from database
                db_status = task_processor.get_task_status(bg_status['final_task_id'])
                if 'error' not in db_status:
                    return {
                        'task_id': task_id,
                        'status': 'done',
                        'progress_percentage': 100,
                        'total_items': db_status.get('total_items', 0),
                        'processed_items': db_status.get('processed_items', 0),
                        'created_at': bg_status.get('created_at', ''),
                        'updated_at': bg_status.get('last_update', ''),
                        'ready_for_download': True
                    }
            
            # Return background status
            status_map = {
                'pending': 'pending',
                'processing': 'processing', 
                'completed': 'done',
                'failed': 'failed'
            }
            
            return {
                'task_id': task_id,
                'status': status_map.get(bg_status['status'], bg_status['status']),
                'progress_percentage': bg_status.get('progress', 0),
                'total_items': bg_status.get('total', 0),
                'processed_items': bg_status.get('completed', 0),
                'created_at': bg_status.get('created_at', ''),
                'updated_at': bg_status.get('last_update', ''),
                'error': bg_status.get('error'),
                'ready_for_download': bg_status['status'] == 'completed'
            }
        
        # Check database for task status
        db_status = task_processor.get_task_status(task_id)
        
        if 'error' in db_status:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Map internal status to API status
        status_map = {
            'pending': 'pending',
            'processing': 'processing',
            'completed': 'done',
            'completed_with_errors': 'done',
            'failed': 'failed'
        }
        
        return {
            'task_id': task_id,
            'status': status_map.get(db_status['status'], db_status['status']),
            'progress_percentage': db_status.get('progress_percentage', 0),
            'total_items': db_status.get('total_items', 0),
            'processed_items': db_status.get('processed_items', 0),
            'created_at': db_status.get('created_at', ''),
            'updated_at': db_status.get('updated_at', ''),
            'ready_for_download': db_status['status'] in ['completed', 'completed_with_errors']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{task_id}")
async def download_results(task_id: str, task: str = "summarize", data_case: str = "external_loss"):
    """
    Download processed results.
    
    Args:
        task_id: Task ID
        task: Task name (default: summarize)
        data_case: Data case (default: external_loss)
        
    Returns:
        CSV file with results
    """
    try:
        # Check if task is completed
        status_info = await get_task_status(task_id)
        
        if status_info['status'] != 'done':
            raise HTTPException(status_code=400, detail="Task not completed yet")
        
        # Get original file path from database
        # First check background task status for final_task_id
        actual_task_id = task_id
        if task_id in background_tasks_status:
            bg_status = background_tasks_status[task_id]
            if 'final_task_id' in bg_status:
                actual_task_id = bg_status['final_task_id']
        
        # Get task details from database to find original file path
        db_status = task_processor.get_task_status(actual_task_id)
        
        if 'error' in db_status or 'original_file_path' not in db_status:
            raise HTTPException(status_code=404, detail="Original file path not found in database")
        
        original_file_path = db_status['original_file_path']
        
        # Verify original file still exists
        if not os.path.exists(original_file_path):
            raise HTTPException(status_code=404, detail="Original uploaded file no longer exists")
        
        # Export results
        export_path = task_processor.export_results(original_file_path, actual_task_id, task, data_case)
        
        if not os.path.exists(export_path):
            raise HTTPException(status_code=404, detail="Results file not found")
        
        return FileResponse(
            path=export_path,
            filename=f"task_{task_id}_results.csv",
            media_type="text/csv"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cleanup/{task_id}")
async def cleanup_task(task_id: str):
    """Clean up task data and temporary files."""
    try:
        # Get task details to find original file
        actual_task_id = task_id
        if task_id in background_tasks_status:
            bg_status = background_tasks_status[task_id]
            if 'final_task_id' in bg_status:
                actual_task_id = bg_status['final_task_id']
        
        db_status = task_processor.get_task_status(actual_task_id)
        
        # Clean up original uploaded file
        if db_status and 'original_file_path' in db_status:
            original_file_path = db_status['original_file_path']
            try:
                if os.path.exists(original_file_path):
                    os.remove(original_file_path)
                    logger.info(f"Cleaned up original file: {original_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up original file {original_file_path}: {str(e)}")
        
        # Remove from background status
        if task_id in background_tasks_status:
            del background_tasks_status[task_id]
        
        # Clean up export files
        export_pattern = f"exports/task_{task_id}_*"
        import glob
        for file_path in glob.glob(export_pattern):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up export file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up {file_path}: {str(e)}")
        
        return {"message": "Task cleaned up successfully"}
        
    except Exception as e:
        logger.error(f"Failed to cleanup task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set")
        exit(1)
    
    # Run the server
    uvicorn.run(
        "api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )