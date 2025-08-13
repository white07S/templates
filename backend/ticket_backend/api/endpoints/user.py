from fastapi import APIRouter, HTTPException
from ticket_backend.models.user import UserCreate, UserRemove, APIResponse
from ticket_backend.auth.authentication import auth_manager

router = APIRouter()

@router.post("/add-user", response_model=APIResponse)
async def add_user(user_data: UserCreate):
    try:
        success = auth_manager.add_user(user_data)
        if success:
            return APIResponse(status="success", message="User added successfully")
        else:
            raise HTTPException(status_code=400, detail="Failed to add user")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.delete("/remove-user", response_model=APIResponse)
async def remove_user(user_data: UserRemove):
    try:
        success = auth_manager.remove_user(user_data.username)
        if success:
            return APIResponse(status="success", message="User removed successfully")
        else:
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")