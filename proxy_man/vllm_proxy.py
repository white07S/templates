#!/usr/bin/env python3
"""
Professional vLLM Proxy Service
A robust, production-ready proxy for vLLM with comprehensive error handling,
logging, rate limiting, and health monitoring.
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Optional, Any, List
from urllib.parse import urljoin

import httpx
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vllm_proxy.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("vllm_proxy")

# Configuration
class Config:
    """Central configuration for the proxy service"""
    # vLLM Backend Configuration
    VLLM_HOST = os.getenv("VLLM_HOST", "127.0.0.1")
    VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
    VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}"
    
    # Proxy Configuration
    PROXY_HOST = os.getenv("PROXY_HOST", "0.0.0.0")
    PROXY_PORT = int(os.getenv("PROXY_PORT", "9000"))
    
    # Model Configuration
    MODEL_NAME = os.getenv("VLLM_MODEL", "Qwen/Qwen-32B-Chat")
    TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "2"))
    DTYPE = os.getenv("DTYPE", "bfloat16")
    GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))
    
    # Rate Limiting
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    # Timeouts
    STARTUP_TIMEOUT = int(os.getenv("STARTUP_TIMEOUT", "300"))  # 5 minutes
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))
    HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
    
    # Retry Configuration
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))

# Simple rate limiter implementation without external dependencies
class RateLimiter:
    """Simple token bucket rate limiter"""
    def __init__(self, rate: int = 60, per: int = 60):
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.time()
        self._lock = asyncio.Lock()
    
    async def check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limit"""
        async with self._lock:
            current = time.time()
            time_passed = current - self.last_check
            self.last_check = current
            self.allowance += time_passed * (self.rate / self.per)
            
            if self.allowance > self.rate:
                self.allowance = self.rate
            
            if self.allowance < 1.0:
                return False
            
            self.allowance -= 1.0
            return True

# Global instances
rate_limiter = RateLimiter(Config.RATE_LIMIT_PER_MINUTE, 60)
vllm_process: Optional[subprocess.Popen] = None
http_client: Optional[httpx.AsyncClient] = None
executor = ThreadPoolExecutor(max_workers=2)

# Health metrics
class HealthMetrics:
    """Track health metrics for monitoring"""
    def __init__(self):
        self.total_requests = 0
        self.failed_requests = 0
        self.last_health_check = None
        self.is_healthy = False
        self.startup_time = None
        self.error_details = None

health_metrics = HealthMetrics()

def get_vllm_command() -> List[str]:
    """Construct vLLM server command with all necessary parameters"""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", Config.MODEL_NAME,
        "--tensor-parallel-size", str(Config.TENSOR_PARALLEL_SIZE),
        "--dtype", Config.DTYPE,
        "--gpu-memory-utilization", str(Config.GPU_MEMORY_UTILIZATION),
        "--host", Config.VLLM_HOST,
        "--port", str(Config.VLLM_PORT),
        "--served-model-name", Config.MODEL_NAME.split("/")[-1],
    ]
    
    # Add optional parameters based on configuration
    if os.getenv("ENABLE_CHUNKED_PREFILL", "true").lower() == "true":
        cmd.append("--enable-chunked-prefill")
    
    if os.getenv("ENABLE_PREFIX_CACHING", "true").lower() == "true":
        cmd.append("--enable-prefix-caching")
    
    if api_key := os.getenv("VLLM_API_KEY"):
        cmd.extend(["--api-key", api_key])
    
    return cmd

async def start_vllm_server():
    """Start the vLLM backend server"""
    global vllm_process
    
    logger.info("Starting vLLM server...")
    logger.info(f"Model: {Config.MODEL_NAME}")
    logger.info(f"Tensor Parallel Size: {Config.TENSOR_PARALLEL_SIZE}")
    
    try:
        cmd = get_vllm_command()
        logger.debug(f"Command: {' '.join(cmd)}")
        
        vllm_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Start logging threads
        executor.submit(log_subprocess_output, vllm_process.stdout, "vLLM-stdout")
        executor.submit(log_subprocess_output, vllm_process.stderr, "vLLM-stderr")
        
        # Wait for vLLM to be ready
        start_time = time.time()
        while time.time() - start_time < Config.STARTUP_TIMEOUT:
            if vllm_process.poll() is not None:
                raise RuntimeError(f"vLLM process exited with code {vllm_process.returncode}")
            
            if await check_vllm_health():
                health_metrics.startup_time = time.time() - start_time
                logger.info(f"✅ vLLM server ready! (took {health_metrics.startup_time:.2f}s)")
                health_metrics.is_healthy = True
                return
            
            await asyncio.sleep(1)
        
        raise TimeoutError(f"vLLM server failed to start within {Config.STARTUP_TIMEOUT} seconds")
        
    except Exception as e:
        logger.error(f"Failed to start vLLM server: {e}")
        health_metrics.error_details = str(e)
        if vllm_process:
            vllm_process.terminate()
            vllm_process = None
        raise

def log_subprocess_output(pipe, prefix: str):
    """Log subprocess output line by line"""
    try:
        for line in pipe:
            if line.strip():
                logger.info(f"[{prefix}] {line.strip()}")
    except Exception as e:
        logger.error(f"Error reading {prefix}: {e}")

async def check_vllm_health() -> bool:
    """Check if vLLM server is healthy"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{Config.VLLM_BASE_URL}/health",
                timeout=2.0
            )
            return response.status_code == 200
    except Exception:
        return False

async def stop_vllm_server():
    """Gracefully stop the vLLM server"""
    global vllm_process
    
    if vllm_process:
        logger.info("Stopping vLLM server...")
        vllm_process.terminate()
        
        try:
            vllm_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("vLLM server didn't stop gracefully, forcing...")
            vllm_process.kill()
            vllm_process.wait()
        
        vllm_process = None
        logger.info("vLLM server stopped")

# FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global http_client
    
    # Startup
    logger.info("Starting vLLM proxy service...")
    
    try:
        # Initialize HTTP client
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(Config.REQUEST_TIMEOUT),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        
        # Start vLLM server
        await start_vllm_server()
        
        # Start background health checker
        asyncio.create_task(periodic_health_check())
        
        logger.info(f"Proxy service ready on http://{Config.PROXY_HOST}:{Config.PROXY_PORT}")
        
    except Exception as e:
        logger.error(f"Failed to start services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down proxy service...")
    
    if http_client:
        await http_client.aclose()
    
    await stop_vllm_server()
    executor.shutdown(wait=True)
    
    logger.info("Proxy service stopped")

# Create FastAPI app
app = FastAPI(
    title="vLLM Proxy Service",
    description="Production-ready proxy for vLLM with comprehensive features",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for logging and metrics
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and update metrics"""
    start_time = time.time()
    request_id = request.headers.get("X-Request-Id", f"req-{int(time.time()*1000)}")
    
    # Add request ID to logs
    logger.info(f"[{request_id}] {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"- {response.status_code} - {duration:.3f}s"
        )
        
        health_metrics.total_requests += 1
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.3f}"
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        health_metrics.failed_requests += 1
        
        logger.error(
            f"[{request_id}] {request.method} {request.url.path} "
            f"- ERROR - {duration:.3f}s - {str(e)}"
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "request_id": request_id,
                "detail": str(e) if os.getenv("DEBUG", "false").lower() == "true" else None
            }
        )

# Rate limiting decorator
async def check_rate_limit(request: Request):
    """Check rate limit for the request"""
    if not Config.RATE_LIMIT_ENABLED:
        return
    
    # Use IP address as identifier
    client_ip = request.client.host if request.client else "unknown"
    
    if not await rate_limiter.check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )

# Health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint with detailed status"""
    is_healthy = await check_vllm_health()
    health_metrics.is_healthy = is_healthy
    health_metrics.last_health_check = datetime.utcnow().isoformat()
    
    status_code = status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": health_metrics.last_health_check,
            "metrics": {
                "total_requests": health_metrics.total_requests,
                "failed_requests": health_metrics.failed_requests,
                "error_rate": health_metrics.failed_requests / max(health_metrics.total_requests, 1),
                "startup_time": health_metrics.startup_time,
                "backend_healthy": is_healthy
            },
            "error": health_metrics.error_details if not is_healthy else None
        }
    )

# Version endpoint
@app.get("/version")
async def version():
    """Get proxy version information"""
    return {
        "proxy_version": "1.0.0",
        "backend_url": Config.VLLM_BASE_URL,
        "model": Config.MODEL_NAME,
        "features": {
            "rate_limiting": Config.RATE_LIMIT_ENABLED,
            "rate_limit_per_minute": Config.RATE_LIMIT_PER_MINUTE if Config.RATE_LIMIT_ENABLED else None,
            "tensor_parallel_size": Config.TENSOR_PARALLEL_SIZE,
            "dtype": Config.DTYPE
        }
    }

# Generic proxy function with retry logic
async def proxy_request(
    request: Request,
    path: str,
    method: str = "POST",
    stream: bool = False
) -> Response:
    """Generic function to proxy requests to vLLM backend with retry logic"""
    
    # Rate limiting
    await check_rate_limit(request)
    
    # Check backend health
    if not health_metrics.is_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Backend service is not available"
        )
    
    # Prepare request
    url = urljoin(Config.VLLM_BASE_URL, path)
    headers = dict(request.headers)
    headers.pop("host", None)  # Remove host header
    
    # Get request body
    body = None
    if method in ["POST", "PUT", "PATCH"]:
        body = await request.body()
    
    # Retry logic
    last_error = None
    for attempt in range(Config.MAX_RETRIES):
        try:
            if stream:
                # Handle streaming responses
                return await handle_streaming_response(url, method, headers, body)
            else:
                # Handle regular responses
                response = await http_client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    content=body,
                    params=request.query_params
                )
                
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.headers.get("content-type", "application/json")
                )
                
        except httpx.TimeoutException as e:
            last_error = e
            logger.warning(f"Request timeout (attempt {attempt + 1}/{Config.MAX_RETRIES}): {e}")
            
        except Exception as e:
            last_error = e
            logger.error(f"Request failed (attempt {attempt + 1}/{Config.MAX_RETRIES}): {e}")
        
        if attempt < Config.MAX_RETRIES - 1:
            await asyncio.sleep(Config.RETRY_DELAY * (attempt + 1))
    
    # All retries failed
    health_metrics.failed_requests += 1
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=f"Failed to proxy request after {Config.MAX_RETRIES} attempts: {str(last_error)}"
    )

async def handle_streaming_response(url: str, method: str, headers: dict, body: bytes) -> StreamingResponse:
    """Handle streaming responses from vLLM"""
    async def stream_generator():
        try:
            async with http_client.stream(
                method=method,
                url=url,
                headers=headers,
                content=body
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield json.dumps({"error": str(e)}).encode()
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )

# OpenAI-compatible endpoints
@app.post("/v1/completions")
async def completions(request: Request):
    """Proxy completions endpoint"""
    # Check if streaming is requested
    try:
        body = await request.json()
        stream = body.get("stream", False)
    except:
        stream = False
    
    return await proxy_request(request, "/v1/completions", stream=stream)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Proxy chat completions endpoint"""
    # Check if streaming is requested
    try:
        body = await request.json()
        stream = body.get("stream", False)
    except:
        stream = False
    
    return await proxy_request(request, "/v1/chat/completions", stream=stream)

@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """Proxy embeddings endpoint"""
    return await proxy_request(request, "/v1/embeddings")

@app.get("/v1/models")
async def models(request: Request):
    """Proxy models endpoint"""
    return await proxy_request(request, "/v1/models", method="GET")

@app.post("/v1/audio/transcriptions")
async def transcriptions(request: Request):
    """Proxy audio transcriptions endpoint"""
    return await proxy_request(request, "/v1/audio/transcriptions")

# vLLM-specific endpoints
@app.post("/tokenize")
async def tokenize(request: Request):
    """Proxy tokenize endpoint"""
    return await proxy_request(request, "/tokenize")

@app.post("/detokenize")
async def detokenize(request: Request):
    """Proxy detokenize endpoint"""
    return await proxy_request(request, "/detokenize")

@app.post("/pooling")
async def pooling(request: Request):
    """Proxy pooling endpoint"""
    return await proxy_request(request, "/pooling")

@app.post("/score")
async def score(request: Request):
    """Proxy score endpoint"""
    return await proxy_request(request, "/score")

@app.post("/rerank")
async def rerank(request: Request):
    """Proxy rerank endpoint"""
    return await proxy_request(request, "/rerank")

@app.post("/v1/rerank")
async def v1_rerank(request: Request):
    """Proxy v1 rerank endpoint"""
    return await proxy_request(request, "/v1/rerank")

@app.post("/v2/rerank")
async def v2_rerank(request: Request):
    """Proxy v2 rerank endpoint"""
    return await proxy_request(request, "/v2/rerank")

# Metrics endpoint (if available)
@app.get("/metrics")
async def metrics(request: Request):
    """Proxy metrics endpoint"""
    return await proxy_request(request, "/metrics", method="GET")

# Ping endpoint
@app.get("/ping")
async def ping():
    """Simple ping endpoint"""
    return {"status": "pong", "timestamp": datetime.utcnow().isoformat()}

# Background health checker
async def periodic_health_check():
    """Periodically check vLLM health"""
    while True:
        try:
            await asyncio.sleep(Config.HEALTH_CHECK_INTERVAL)
            is_healthy = await check_vllm_health()
            
            if not is_healthy and health_metrics.is_healthy:
                logger.error("vLLM backend became unhealthy!")
                health_metrics.error_details = "Backend health check failed"
            elif is_healthy and not health_metrics.is_healthy:
                logger.info("vLLM backend recovered!")
                health_metrics.error_details = None
            
            health_metrics.is_healthy = is_healthy
            
        except Exception as e:
            logger.error(f"Health check error: {e}")

# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    asyncio.create_task(stop_vllm_server())
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Main entry point
if __name__ == "__main__":
    try:
        # Configure uvicorn
        uvicorn.run(
            app,
            host=Config.PROXY_HOST,
            port=Config.PROXY_PORT,
            log_level="info",
            access_log=True,
            workers=1  # Single worker to maintain state
        )
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Ensure cleanup
        asyncio.run(stop_vllm_server())