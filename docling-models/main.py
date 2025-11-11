from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import hashlib
import random
import string
import asyncio
from datetime import datetime, timezone
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NFRF Connect")

# Mount static files for assets
app.mount("/asset", StaticFiles(directory="asset"), name="asset")

# Global state for the background process
process_status = "pending"
process_running = False

# Mock function implementations
def start_endpoint():
    """Mock function to start endpoint - returns a hash"""
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
    hash_object = hashlib.sha256(random_string.encode())
    return hash_object.hexdigest()

def stop_endpoint():
    """Mock function to stop endpoint - returns a hash"""
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
    hash_object = hashlib.sha256(random_string.encode())
    return hash_object.hexdigest()

def status_endpoint():
    """Mock function to get status - returns running, pending, or failed"""
    return process_status

def main_app_url():
    """Mock function to get main app URL"""
    return "https://app.nfrf-connect.com/dashboard"

# Background process functions
async def start_background_process():
    """Start the background process"""
    global process_status, process_running
    try:
        logger.info("Starting background process...")
        hash_result = start_endpoint()
        process_status = "running"
        process_running = True
        logger.info(f"Background process started successfully. Hash: {hash_result}")
    except Exception as e:
        process_status = "failed"
        process_running = False
        logger.error(f"Failed to start background process: {e}")

async def stop_background_process():
    """Stop the background process"""
    global process_status, process_running
    try:
        logger.info("Stopping background process...")
        hash_result = stop_endpoint()
        process_status = "pending"
        process_running = False
        logger.info(f"Background process stopped successfully. Hash: {hash_result}")
    except Exception as e:
        process_status = "failed"
        logger.error(f"Failed to stop background process: {e}")

# Initialize scheduler
scheduler = AsyncIOScheduler()

# Schedule start at 6am UTC
scheduler.add_job(
    start_background_process,
    trigger=CronTrigger(hour=6, minute=0, timezone=timezone.utc),
    id='start_job',
    replace_existing=True
)

# Schedule stop at 6pm UTC
scheduler.add_job(
    stop_background_process,
    trigger=CronTrigger(hour=18, minute=0, timezone=timezone.utc),
    id='stop_job',
    replace_existing=True
)

@app.on_event("startup")
async def startup_event():
    """Start the scheduler on app startup"""
    scheduler.start()
    logger.info("Scheduler started. Background process will run from 6am to 6pm UTC.")

    # Check if we should be running now
    current_hour = datetime.now(timezone.utc).hour
    if 6 <= current_hour < 18:
        await start_background_process()
    else:
        logger.info("Outside operational hours (6am-6pm UTC). Process is pending.")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown the scheduler"""
    scheduler.shutdown()
    logger.info("Scheduler shutdown.")

# API Endpoints
@app.get("/api/status")
async def get_status():
    """Get current process status"""
    return {
        "status": status_endpoint(),
        "is_running": process_running,
        "current_time": datetime.now(timezone.utc).isoformat(),
        "operational_hours": "6:00 AM - 6:00 PM UTC"
    }

@app.get("/api/app-url")
async def get_app_url():
    """Get main application URL"""
    return {"url": main_app_url()}

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main SPA page"""
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)