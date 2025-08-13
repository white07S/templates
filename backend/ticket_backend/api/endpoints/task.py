from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
import os
import uuid
import json
import glob
from typing import List
from datetime import datetime
from ticket_backend.models.task import TaskSubmit, TaskResponse, TaskStatusResponse, AvailableTasksResponse, TaskStatus, UserTasksResponse, UserTaskInfo
from ticket_backend.config.offerings import get_available_data_types, is_valid_data_type, is_valid_task, get_mandatory_columns
from ticket_backend.utils.file_handler import save_uploaded_file, validate_file_columns

router = APIRouter()

@router.get("/available-tasks", response_model=AvailableTasksResponse)
async def get_available_tasks():
    return AvailableTasksResponse(data_types=get_available_data_types())

@router.post("/submit-task", response_model=TaskResponse)
async def submit_task(
    username: str = Form(...),
    data_type: str = Form(...),
    tasks: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        tasks_list = [task.strip() for task in tasks.split(",")]
        
        if not is_valid_data_type(data_type):
            raise HTTPException(status_code=400, detail="Invalid data type")
        
        for task in tasks_list:
            if not is_valid_task(data_type, task):
                raise HTTPException(status_code=400, detail=f"Invalid task '{task}' for data type '{data_type}'")
        
        if not file.filename.endswith(('.xlsx', '.csv')):
            raise HTTPException(status_code=400, detail="File must be XLSX or CSV format")
        
        task_id = str(uuid.uuid4())
        task_dir = f"data/tasks/{task_id}"
        os.makedirs(task_dir, exist_ok=True)
        
        file_path = await save_uploaded_file(file, task_dir)
        
        mandatory_columns = get_mandatory_columns(data_type)
        if not validate_file_columns(file_path, mandatory_columns):
            raise HTTPException(status_code=400, detail=f"File must contain mandatory columns: {mandatory_columns}")
        
        task_info = {
            "task_id": task_id,
            "username": username,
            "data_type": data_type,
            "tasks": tasks_list,
            "file_path": file_path,
            "status": "pending",
            "submitted_at": datetime.now().isoformat()
        }
        
        with open(f"{task_dir}/task_info.json", "w") as f:
            import json
            json.dump(task_info, f, indent=2)
        
        return TaskResponse(task_id=task_id, status=TaskStatus.PENDING)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/task-status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    try:
        task_dir = f"data/tasks/{task_id}"
        if not os.path.exists(task_dir):
            raise HTTPException(status_code=404, detail="Task not found")
        
        completed_dir = f"{task_dir}/completed"
        status = TaskStatus.COMPLETED if os.path.exists(completed_dir) else TaskStatus.PENDING
        
        return TaskStatusResponse(task_id=task_id, status=status)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/task-result/{task_id}")
async def get_task_result(task_id: str):
    try:
        task_dir = f"data/tasks/{task_id}"
        completed_dir = f"{task_dir}/completed"
        
        if not os.path.exists(task_dir):
            raise HTTPException(status_code=404, detail="Task not found")
        
        if not os.path.exists(completed_dir):
            raise HTTPException(status_code=400, detail="Task is still pending")
        
        result_files = [f for f in os.listdir(completed_dir) if f.endswith(('.xlsx', '.csv'))]
        if not result_files:
            raise HTTPException(status_code=404, detail="Result file not found")
        
        result_file_path = os.path.join(completed_dir, result_files[0])
        return FileResponse(result_file_path, filename=result_files[0])
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/user-tasks", response_model=UserTasksResponse)
async def get_user_tasks(
    username: str = Form(...)
):
    try:
        
        user_tasks = []
        tasks_dir = "data/tasks"
        
        if not os.path.exists(tasks_dir):
            return UserTasksResponse(tasks=[])
        
        # Iterate through all task directories
        for task_dir in glob.glob(f"{tasks_dir}/*"):
            if os.path.isdir(task_dir):
                task_info_file = os.path.join(task_dir, "task_info.json")
                
                if os.path.exists(task_info_file):
                    try:
                        with open(task_info_file, 'r') as f:
                            task_info = json.load(f)
                        
                        # Check if this task belongs to the requesting user
                        if task_info.get('username') == username:
                            # Determine current status
                            completed_dir = os.path.join(task_dir, "completed")
                            status = TaskStatus.COMPLETED if os.path.exists(completed_dir) else TaskStatus.PENDING
                            
                            # Get file name from the original file path
                            file_path = task_info.get('file_path', '')
                            file_name = os.path.basename(file_path) if file_path else 'unknown'
                            
                            # Create datetime object (use current time if not stored)
                            submitted_at = datetime.now()
                            if 'submitted_at' in task_info:
                                try:
                                    submitted_at = datetime.fromisoformat(task_info['submitted_at'])
                                except:
                                    pass
                            
                            user_task = UserTaskInfo(
                                task_id=task_info.get('task_id', os.path.basename(task_dir)),
                                data_type=task_info.get('data_type', ''),
                                tasks=task_info.get('tasks', []),
                                status=status,
                                submitted_at=submitted_at,
                                file_name=file_name
                            )
                            user_tasks.append(user_task)
                    
                    except json.JSONDecodeError:
                        continue
        
        # Sort by submitted_at descending (newest first)
        user_tasks.sort(key=lambda x: x.submitted_at, reverse=True)
        
        return UserTasksResponse(tasks=user_tasks)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")