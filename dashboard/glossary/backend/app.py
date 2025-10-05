from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from glossary_router import router as glossary_router

app = FastAPI(title="Glossary Management API")

# Configure CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(glossary_router)

@app.get("/")
async def root():
    return {"message": "Glossary Management API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)