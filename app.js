import logging
import sys
import time
import uuid
import asyncio
import json
import queue
import threading
import multiprocessing
from datetime import datetime
from typing import Callable, Optional, Dict, Any, List
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

# ========== Configuration ==========
class LogFormat(Enum):
    JSON = "json"
    PLAIN = "plain"
    CUSTOM = "custom"

class LogConfig:
    """Centralized logging configuration"""
    # Performance
    USE_MULTIPROCESSING = False  # True = separate process, False = thread
    QUEUE_SIZE = 10000  # Max queued log records
    BATCH_SIZE = 100  # Flush logs in batches
    FLUSH_INTERVAL = 0.5  # seconds
    
    # Output destinations
    CONSOLE_ENABLED = True
    CONSOLE_LEVEL = logging.INFO
    FILE_ENABLED = True
    FILE_LEVEL = logging.DEBUG
    FILE_NAME = "app.log"
    FILE_MAX_BYTES = 100_000_000  # 100MB
    FILE_BACKUP_COUNT = 10
    
    # Formatting
    LOG_FORMAT = LogFormat.JSON
    
    # Plain text format
    PLAIN_FORMAT = "[%(asctime)s] [%(levelname)-8s] [%(correlation_id)s] %(name)s - %(message)s"
    
    # Custom format function (if LOG_FORMAT = CUSTOM)
    @staticmethod
    def custom_formatter(record: logging.LogRecord) -> str:
        """Define your own format here"""
        return f"{record.levelname}|{record.getMessage()}|{getattr(record, 'correlation_id', '-')}"
    
    # JSON fields to include
    JSON_FIELDS = [
        "timestamp", "level", "logger", "message", "correlation_id",
        "method", "path", "status_code", "duration_ms", "client_ip",
        "user_id", "error_type", "stack_trace", "thread", "process"
    ]
    
    # Sampling (reduce volume)
    SAMPLE_RATE = 1.0  # 1.0 = log everything, 0.1 = log 10%
    ALWAYS_LOG_ERRORS = True  # Override sampling for errors
    
    # Request/Response body logging (careful with PII!)
    LOG_REQUEST_BODY = False
    LOG_RESPONSE_BODY = False
    MAX_BODY_LENGTH = 1000
    REDACT_PATTERNS = ["password", "token", "secret", "api_key", "authorization"]
    
    # Performance monitoring
    SLOW_REQUEST_THRESHOLD_MS = 1000
    LOG_SLOW_REQUESTS = True

# ========== Custom JSON Formatter ==========
class AdvancedJSONFormatter(logging.Formatter):
    """High-performance JSON formatter with configurable fields"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {}
        
        # Base fields
        if "timestamp" in LogConfig.JSON_FIELDS:
            log_obj["timestamp"] = datetime.utcnow().isoformat() + "Z"
        if "level" in LogConfig.JSON_FIELDS:
            log_obj["level"] = record.levelname
        if "logger" in LogConfig.JSON_FIELDS:
            log_obj["logger"] = record.name
        if "message" in LogConfig.JSON_FIELDS:
            log_obj["message"] = record.getMessage()
        
        # Add all extra fields from record
        for field in LogConfig.JSON_FIELDS:
            if hasattr(record, field) and field not in log_obj:
                value = getattr(record, field)
                if value is not None:
                    log_obj[field] = value
        
        # Add exception info if present
        if record.exc_info and "stack_trace" in LogConfig.JSON_FIELDS:
            log_obj["stack_trace"] = self.formatException(record.exc_info)
        
        # Thread/Process info
        if "thread" in LogConfig.JSON_FIELDS:
            log_obj["thread"] = record.thread
        if "process" in LogConfig.JSON_FIELDS:
            log_obj["process"] = record.process
            
        return json.dumps(log_obj, default=str)

# ========== Non-blocking Queue Handler ==========
class NonBlockingQueueHandler(QueueHandler):
    """Queue handler that never blocks the main thread"""
    
    def enqueue(self, record):
        try:
            # Non-blocking put with timeout
            self.queue.put_nowait(record)
        except queue.Full:
            # Silently drop if queue is full (or implement your overflow strategy)
            pass  # Could increment a dropped_logs counter here

# ========== Background Log Worker ==========
class LogWorker:
    """Background worker that processes logs from queue"""
    
    def __init__(self, log_queue, handlers):
        self.queue = log_queue
        self.handlers = handlers
        self.running = False
        self.stats = {
            "processed": 0,
            "dropped": 0,
            "errors": 0
        }
        
    def process_batch(self):
        """Process a batch of log records"""
        batch = []
        deadline = time.time() + LogConfig.FLUSH_INTERVAL
        
        while time.time() < deadline and len(batch) < LogConfig.BATCH_SIZE:
            try:
                timeout = max(0.01, deadline - time.time())
                record = self.queue.get(timeout=timeout)
                batch.append(record)
            except queue.Empty:
                break
                
        # Process the batch
        for record in batch:
            for handler in self.handlers:
                try:
                    handler.handle(record)
                    self.stats["processed"] += 1
                except Exception:
                    self.stats["errors"] += 1
                    
    def run(self):
        """Main worker loop"""
        self.running = True
        while self.running:
            try:
                self.process_batch()
            except Exception:
                pass  # Worker must never crash
                
    def stop(self):
        self.running = False

# ========== Global Logger Setup ==========
class GlobalLogger:
    """Manages the global non-blocking logging system"""
    
    def __init__(self):
        self.queue = None
        self.worker = None
        self.worker_thread = None
        self.listener = None
        self.initialized = False
        
    def setup(self):
        """Initialize the logging system"""
        if self.initialized:
            return
            
        # Create queue (multiprocessing or threading based on config)
        if LogConfig.USE_MULTIPROCESSING:
            self.queue = multiprocessing.Queue(LogConfig.QUEUE_SIZE)
        else:
            self.queue = queue.Queue(LogConfig.QUEUE_SIZE)
        
        # Create handlers for actual output
        handlers = []
        
        # Console handler
        if LogConfig.CONSOLE_ENABLED:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(LogConfig.CONSOLE_LEVEL)
            handlers.append(console_handler)
        
        # File handler
        if LogConfig.FILE_ENABLED:
            file_handler = RotatingFileHandler(
                LogConfig.FILE_NAME,
                maxBytes=LogConfig.FILE_MAX_BYTES,
                backupCount=LogConfig.FILE_BACKUP_COUNT
            )
            file_handler.setLevel(LogConfig.FILE_LEVEL)
            handlers.append(file_handler)
        
        # Apply formatters
        formatter = self._get_formatter()
        for handler in handlers:
            handler.setFormatter(formatter)
        
        # Create and start the background worker
        if LogConfig.USE_MULTIPROCESSING:
            # Use QueueListener for multiprocessing
            self.listener = QueueListener(self.queue, *handlers, respect_handler_level=True)
            self.listener.start()
        else:
            # Use custom worker for threading (more control)
            self.worker = LogWorker(self.queue, handlers)
            self.worker_thread = threading.Thread(target=self.worker.run, daemon=True)
            self.worker_thread.start()
        
        # Configure root logger to use queue
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Capture everything, filter at handler level
        root_logger.handlers = [NonBlockingQueueHandler(self.queue)]
        
        self.initialized = True
        
    def _get_formatter(self):
        """Get the appropriate formatter based on config"""
        if LogConfig.LOG_FORMAT == LogFormat.JSON:
            return AdvancedJSONFormatter()
        elif LogConfig.LOG_FORMAT == LogFormat.PLAIN:
            return logging.Formatter(LogConfig.PLAIN_FORMAT)
        elif LogConfig.LOG_FORMAT == LogFormat.CUSTOM:
            # Create a custom formatter that uses the config function
            class CustomFormatter(logging.Formatter):
                def format(self, record):
                    return LogConfig.custom_formatter(record)
            return CustomFormatter()
        else:
            return logging.Formatter()
    
    def shutdown(self):
        """Gracefully shutdown the logging system"""
        if LogConfig.USE_MULTIPROCESSING and self.listener:
            self.listener.stop()
        elif self.worker:
            self.worker.stop()
            if self.worker_thread:
                self.worker_thread.join(timeout=2)
        
    def get_stats(self):
        """Get logging statistics"""
        if self.worker:
            return self.worker.stats
        return {}

# ========== Initialize Global Logger ==========
global_logger = GlobalLogger()
global_logger.setup()
log = logging.getLogger("app")

# ========== Context Manager for App Lifecycle ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle"""
    # Startup
    log.info("app.startup", extra={"event": "startup", "config": LogConfig.__dict__})
    
    # Setup asyncio exception handler
    loop = asyncio.get_running_loop()
    def handle_exception(loop, context):
        exception = context.get("exception")
        log.error("asyncio.exception", 
                 extra={"message": context.get("message", "Unknown error")},
                 exc_info=exception)
    loop.set_exception_handler(handle_exception)
    
    yield
    
    # Shutdown
    log.info("app.shutdown", extra={"event": "shutdown", "stats": global_logger.get_stats()})
    global_logger.shutdown()

# ========== FastAPI App ==========
app = FastAPI(title="Advanced Non-Blocking Logger", lifespan=lifespan)

# ========== Correlation ID & Logging Middleware ==========
@app.middleware("http")
async def logging_middleware(request: Request, call_next: Callable):
    """Main logging middleware with all features"""
    start_time = time.time()
    
    # Correlation ID
    correlation_id = request.headers.get("x-correlation-id", str(uuid.uuid4()))
    request.state.correlation_id = correlation_id
    
    # Sampling decision
    should_log = (
        LogConfig.SAMPLE_RATE >= 1.0 or 
        (LogConfig.SAMPLE_RATE > 0 and hash(correlation_id) % int(1/LogConfig.SAMPLE_RATE) == 0)
    )
    
    # Extract request metadata
    client_ip = request.client.host if request.client else None
    user_id = request.headers.get("x-user-id")
    
    # Log request start
    if should_log:
        extra = {
            "correlation_id": correlation_id,
            "method": request.method,
            "path": request.url.path,
            "client_ip": client_ip,
            "user_id": user_id,
            "query_params": dict(request.query_params) if request.query_params else None
        }
        
        # Optional: Log request body (be careful with PII!)
        if LogConfig.LOG_REQUEST_BODY and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                request._body = body  # Cache for endpoint use
                body_str = body.decode()[:LogConfig.MAX_BODY_LENGTH]
                # Redact sensitive fields
                for pattern in LogConfig.REDACT_PATTERNS:
                    if pattern.lower() in body_str.lower():
                        body_str = "[REDACTED]"
                        break
                extra["request_body"] = body_str
            except:
                pass
        
        log.info("http.request.start", extra=extra)
    
    # Process request
    try:
        response = await call_next(request)
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Always log errors regardless of sampling
        log.exception("http.request.error",
                     extra={
                         "correlation_id": correlation_id,
                         "method": request.method,
                         "path": request.url.path,
                         "client_ip": client_ip,
                         "user_id": user_id,
                         "duration_ms": duration_ms,
                         "error_type": type(e).__name__
                     })
        
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "correlation_id": correlation_id}
        )
    
    # Calculate duration
    duration_ms = int((time.time() - start_time) * 1000)
    
    # Log response
    if should_log or (LogConfig.ALWAYS_LOG_ERRORS and response.status_code >= 400):
        extra = {
            "correlation_id": correlation_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "client_ip": client_ip,
            "user_id": user_id
        }
        
        # Log slow requests
        if LogConfig.LOG_SLOW_REQUESTS and duration_ms > LogConfig.SLOW_REQUEST_THRESHOLD_MS:
            extra["slow_request"] = True
            log.warning("http.request.slow", extra=extra)
        else:
            level = logging.ERROR if response.status_code >= 500 else logging.INFO
            log.log(level, "http.request.complete", extra=extra)
    
    # Add correlation ID to response
    response.headers["x-correlation-id"] = correlation_id
    response.headers["x-process-time"] = str(duration_ms)
    
    return response

# ========== Exception Handlers ==========
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    
    log.warning("http.exception",
               extra={
                   "correlation_id": correlation_id,
                   "status_code": exc.status_code,
                   "detail": exc.detail,
                   "path": request.url.path
               })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "correlation_id": correlation_id}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    
    log.exception("app.exception",
                 extra={
                     "correlation_id": correlation_id,
                     "error_type": type(exc).__name__,
                     "path": request.url.path
                 })
    
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "correlation_id": correlation_id}
    )

# ========== Demo Endpoints ==========
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "logger_stats": global_logger.get_stats()}

@app.get("/test")
async def test():
    """Normal endpoint"""
    log.info("business.event", extra={"event": "test_endpoint_accessed"})
    return {"message": "Test successful"}

@app.get("/slow")
async def slow():
    """Slow endpoint to test slow request logging"""
    await asyncio.sleep(1.5)
    return {"message": "Slow operation completed"}

@app.get("/error")
async def error():
    """Endpoint that raises an error"""
    raise ValueError("Intentional error for testing")

@app.post("/echo")
async def echo(request: Request):
    """Echo endpoint to test request body logging"""
    body = await request.body()
    return {"received": body.decode()}

@app.get("/background")
async def background():
    """Test background task error handling"""
    async def failing_task():
        await asyncio.sleep(0.1)
        raise RuntimeError("Background task failed")
    
    asyncio.create_task(failing_task())
    return {"message": "Background task started"}

# ========== Custom Business Logger Example ==========
class BusinessLogger:
    """Example of a domain-specific logger"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(f"business.{name}")
    
    def log_transaction(self, transaction_id: str, amount: float, status: str):
        self.logger.info("transaction",
                        extra={
                            "transaction_id": transaction_id,
                            "amount": amount,
                            "status": status,
                            "timestamp": datetime.utcnow().isoformat()
                        })
    
    def log_audit_event(self, user_id: str, action: str, resource: str):
        self.logger.info("audit",
                        extra={
                            "user_id": user_id,
                            "action": action,
                            "resource": resource,
                            "timestamp": datetime.utcnow().isoformat()
                        })

# Usage example
biz_logger = BusinessLogger("payments")

@app.post("/payment")
async def payment():
    transaction_id = str(uuid.uuid4())
    biz_logger.log_transaction(transaction_id, 99.99, "success")
    return {"transaction_id": transaction_id}

# ========== Run Instructions ==========
"""
To run this server:

1. Install dependencies:
   pip install fastapi uvicorn

2. Run with:
   uvicorn main:app --reload --port 8000

3. Test endpoints:
   - http://localhost:8000/ - Health check with stats
   - http://localhost:8000/test - Normal request
   - http://localhost:8000/slow - Slow request (tests threshold)
   - http://localhost:8000/error - Error handling
   - http://localhost:8000/background - Background task error

4. Check logs:
   - Console: Real-time JSON logs
   - File: app.log (rotating, max 100MB x 10 files)

5. Customize in LogConfig class:
   - Change format (JSON/Plain/Custom)
   - Adjust sampling rate
   - Enable/disable features
   - Set performance thresholds
"""
