#!/usr/bin/env python3
"""
vLLM Reverse Proxy Server
Manages multiple vLLM instances and routes requests based on endpoints.
"""

import asyncio
import os
import sys
import time
import signal
import subprocess
import aiohttp
from aiohttp import web
import logging
import json
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VLLMService:
    """Manages a single vLLM service instance."""
    
    def __init__(self, name: str, command: list, port: int, cuda_device: Optional[str] = None):
        self.name = name
        self.command = command
        self.port = port
        self.cuda_device = cuda_device
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://localhost:{port}"
        self.health_url = f"{self.base_url}/health"
        
    async def start(self):
        """Start the vLLM service process."""
        env = os.environ.copy()
        if self.cuda_device is not None:
            env['CUDA_VISIBLE_DEVICES'] = self.cuda_device
            
        logger.info(f"Starting {self.name} service on port {self.port}...")
        if self.cuda_device:
            logger.info(f"Using CUDA device: {self.cuda_device}")
        logger.info(f"Command: {' '.join(self.command)}")
        
        self.process = subprocess.Popen(
            self.command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Start background tasks to log output
        asyncio.create_task(self._log_output(self.process.stdout, f"{self.name}-stdout"))
        asyncio.create_task(self._log_output(self.process.stderr, f"{self.name}-stderr"))
        
    async def _log_output(self, stream, prefix):
        """Log output from subprocess streams."""
        try:
            for line in stream:
                if line.strip():
                    logger.debug(f"[{prefix}] {line.strip()}")
        except Exception as e:
            logger.error(f"Error reading {prefix}: {e}")
            
    async def wait_until_ready(self, timeout: int = 300, check_interval: int = 5):
        """Wait until the service is ready by checking health endpoint."""
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < timeout:
                try:
                    async with session.get(self.health_url) as response:
                        if response.status == 200:
                            logger.info(f"{self.name} service is ready!")
                            return True
                except Exception as e:
                    logger.debug(f"Health check failed for {self.name}: {e}")
                
                await asyncio.sleep(check_interval)
                
        raise TimeoutError(f"{self.name} service failed to start within {timeout} seconds")
        
    def stop(self):
        """Stop the vLLM service process."""
        if self.process:
            logger.info(f"Stopping {self.name} service...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing {self.name} service...")
                self.process.kill()
            self.process = None

class VLLMReverseProxy:
    """Reverse proxy server that routes requests to appropriate vLLM services."""
    
    def __init__(self, proxy_port: int = 8888):
        self.proxy_port = proxy_port
        self.services: Dict[str, VLLMService] = {}
        self.app = web.Application()
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup proxy routes."""
        # Add catch-all route
        self.app.router.add_route('*', '/{path:.*}', self.proxy_handler)
        
    def add_service(self, service: VLLMService, route_prefix: str):
        """Add a vLLM service with its route prefix."""
        self.services[route_prefix] = service
        
    def _determine_target_service(self, path: str) -> Optional[VLLMService]:
        """Determine which service should handle the request based on path."""
        # Route logic based on endpoint patterns
        if path.startswith('/v1/embeddings'):
            return self.services.get('embeddings')
        elif path.startswith('/v1/completions') or path.startswith('/v1/chat/completions'):
            return self.services.get('reasoning')
        elif path == '/health':
            # Return health check for proxy itself
            return None
        elif path.startswith('/v1/models'):
            # Models endpoint - route to reasoning by default
            return self.services.get('reasoning')
        else:
            # Default to reasoning service for other endpoints
            return self.services.get('reasoning')
            
    async def proxy_handler(self, request: web.Request) -> web.Response:
        """Handle incoming requests and proxy to appropriate service."""
        path = request.path_qs
        
        # Special handling for proxy health check
        if path == '/health':
            health_status = {
                'status': 'healthy',
                'services': {}
            }
            
            # Check health of all services
            async with aiohttp.ClientSession() as session:
                for name, service in self.services.items():
                    try:
                        async with session.get(service.health_url) as resp:
                            health_status['services'][name] = {
                                'status': 'healthy' if resp.status == 200 else 'unhealthy',
                                'port': service.port
                            }
                    except Exception:
                        health_status['services'][name] = {
                            'status': 'unhealthy',
                            'port': service.port
                        }
                        
            return web.json_response(health_status)
        
        # Determine target service
        target_service = self._determine_target_service(path)
        
        if not target_service:
            return web.json_response(
                {'error': 'No service available for this endpoint'},
                status=404
            )
            
        # Build target URL
        target_url = f"{target_service.base_url}{path}"
        
        # Prepare headers
        headers = dict(request.headers)
        headers.pop('Host', None)
        headers.pop('Content-Length', None)
        
        # Read request body
        body = await request.read()
        
        logger.info(f"Proxying {request.method} {path} to {target_service.name} service")
        
        try:
            # Make request to target service
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    data=body,
                    allow_redirects=False
                ) as response:
                    # Prepare response headers
                    resp_headers = dict(response.headers)
                    resp_headers.pop('Content-Encoding', None)
                    resp_headers.pop('Content-Length', None)
                    resp_headers.pop('Transfer-Encoding', None)
                    
                    # Stream response body
                    body_bytes = await response.read()
                    
                    return web.Response(
                        body=body_bytes,
                        status=response.status,
                        headers=resp_headers
                    )
                    
        except Exception as e:
            logger.error(f"Proxy error: {e}")
            return web.json_response(
                {'error': f'Proxy error: {str(e)}'},
                status=502
            )
            
    async def start_services_sequentially(self, reasoning_service: VLLMService, embeddings_service: VLLMService):
        """Start services sequentially - embeddings only after reasoning is ready."""
        # Start reasoning service first
        logger.info("Starting reasoning service first...")
        await reasoning_service.start()
        await reasoning_service.wait_until_ready()
        
        # Only start embeddings service after reasoning is ready
        logger.info("Reasoning service is ready. Now starting embeddings service...")
        await embeddings_service.start()
        await embeddings_service.wait_until_ready()
        
        logger.info("All services are ready!")
        
    async def start(self, reasoning_service: VLLMService, embeddings_service: VLLMService):
        """Start services sequentially and then the proxy server."""
        # Start services sequentially
        await self.start_services_sequentially(reasoning_service, embeddings_service)
        
        # Start proxy server
        logger.info(f"Starting reverse proxy server on port {self.proxy_port}...")
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.proxy_port)
        await site.start()
        
        logger.info(f"Reverse proxy server is running on http://0.0.0.0:{self.proxy_port}")
        logger.info("Available endpoints:")
        logger.info("  - /v1/completions - Text generation (reasoning model)")
        logger.info("  - /v1/chat/completions - Chat completions (reasoning model)")
        logger.info("  - /v1/embeddings - Text embeddings")
        logger.info("  - /v1/models - List available models")
        logger.info("  - /health - Health check for proxy and services")
        
    def stop(self):
        """Stop all services."""
        for service in self.services.values():
            service.stop()

async def main():
    """Main function to set up and run the proxy server."""
    
    # Configuration
    REASONING_MODEL_PATH = "/model-location"  # Update this path
    EMBEDDINGS_MODEL_PATH = "/models-location"  # Update this path
    REASONING_PORT = 7001
    EMBEDDINGS_PORT = 7002
    PROXY_PORT = 8888
    
    # Optional: Additional vLLM arguments
    REASONING_EXTRA_ARGS = [
        # Add any additional arguments for reasoning model here
        # "--max-model-len", "8192",
        # "--gpu-memory-utilization", "0.9",
    ]
    
    EMBEDDINGS_EXTRA_ARGS = [
        # Add any additional arguments for embeddings model here
        # "--max-model-len", "512",
        # "--gpu-memory-utilization", "0.5",
    ]
    
    # Create proxy server
    proxy = VLLMReverseProxy(proxy_port=PROXY_PORT)
    
    # Create reasoning service (GPU 0)
    reasoning_command = [
        "vllm", "serve",
        REASONING_MODEL_PATH,
        "--reasoning-parser", "deepseek_r1",
        "--tensor-parallel-size", "1",
        "--port", str(REASONING_PORT),
        "--host", "0.0.0.0"
    ] + REASONING_EXTRA_ARGS
    
    reasoning_service = VLLMService(
        name="reasoning",
        command=reasoning_command,
        port=REASONING_PORT,
        cuda_device="0"  # Use first GPU
    )
    
    # Create embeddings service (GPU 1)
    embeddings_command = [
        "vllm", "serve",
        EMBEDDINGS_MODEL_PATH,
        "--task", "embed",  # Specify embeddings task
        "--tensor-parallel-size", "1",
        "--port", str(EMBEDDINGS_PORT),
        "--host", "0.0.0.0"
    ] + EMBEDDINGS_EXTRA_ARGS
    
    embeddings_service = VLLMService(
        name="embeddings",
        command=embeddings_command,
        port=EMBEDDINGS_PORT,
        cuda_device="1"  # Use second GPU
    )
    
    # Add services to proxy
    proxy.add_service(reasoning_service, 'reasoning')
    proxy.add_service(embeddings_service, 'embeddings')
    
    # Handle shutdown gracefully
    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        proxy.stop()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start everything (services will start sequentially)
        await proxy.start(reasoning_service, embeddings_service)
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        proxy.stop()
        sys.exit(1)

if __name__ == "__main__":
    # Check if vllm command is available
    try:
        subprocess.run(["vllm", "--help"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("vLLM command not found. Please ensure vLLM is installed and in PATH.")
        logger.error("Install with: pip install vllm")
        sys.exit(1)
        
    asyncio.run(main())
