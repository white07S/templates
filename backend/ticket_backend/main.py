from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ticket_backend.api.endpoints import task

app = FastAPI(
    title="Task Ticket Platform",
    description="A platform for processing various types of data through analytical tasks",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(task.router, prefix="/api", tags=["tasks"])

@app.get("/")
async def root():
    return {"message": "Task Ticket Platform API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)