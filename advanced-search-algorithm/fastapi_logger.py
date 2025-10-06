"""
Advanced Observability Router for FastAPI
Complete plug-and-play solution for logging, error handling, and monitoring
"""
import json
import sys
import time
import traceback
import uuid
import psutil
import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, List, Union
from contextvars import ContextVar
from collections import defaultdict
from functools import wraps
import logging
import logging.handlers
from pathlib import Path

from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, ValidationError

try:
    from config import config, PERFORMANCE_THRESHOLDS, STATUS_CODE_CATEGORIES
except ImportError:
    # Fallback if config.py is not available
    class ConfigFallback:
        app_name = "fastapi-app"
        environment = "development"
        log_level = "INFO"
        log_format = "json"
        enable_correlation_id = True
        correlation_id_header = "X-Correlation-ID"
        mask_sensitive_data = True
        sensitive_fields = ["password", "token", "api_key"]
        mask_pattern = "****"
        excluded_paths = ["/health", "/metrics"]
        max_body_log_size = 10000
        log_request_body = True
        log_response_body = True
        capture_stack_traces = True
        enable_performance_logging = True
        slow_request_threshold_ms = 1000
        enable_memory_logging = True
        enable_cpu_logging = True
        log_to_console = True
        log_to_file = False
        
        def should_log_path(self, path):
            return not any(path.startswith(excluded) for excluded in self.excluded_paths)
        
        def should_mask_field(self, field_name):
            return any(s in field_name.lower() for s in self.sensitive_fields)
        
        def get_masked_value(self, value):
            return self.mask_pattern
    
    config = ConfigFallback()
    PERFORMANCE_THRESHOLDS = {"fast": 100, "normal": 500, "slow": 1000, "critical": 5000}
    STATUS_CODE_CATEGORIES = {
        "success": range(200, 300),
        "client_error": range(400, 500),
        "server_error": range(500, 600)
    }


# Context variables for request-scoped data
correlation_id_context: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
trace_id_context: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)
span_id_context: ContextVar[Optional[str]] = ContextVar('span_id', default=None)
request_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar('request_context', default={})


class StructuredLogger:
    """Advanced structured logger with JSON output and context management"""
    
    def __init__(self, name: str = "fastapi"):
        self.logger = logging.getLogger(name)
        self._setup_logger()
        self.error_counter = defaultdict(int)
        self.last_error_time = defaultdict(float)
    
    def _setup_logger(self):
        """Configure the logger with handlers and formatters"""
        self.logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))
        self.logger.handlers = []
        
        if config.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self._get_formatter())
            self.logger.addHandler(console_handler)
        
        if config.log_to_file and config.log_file_path:
            file_path = Path(config.log_file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=self._parse_size(config.log_rotation_size),
                backupCount=config.log_retention_days
            )
            file_handler.setFormatter(self._get_formatter())
            self.logger.addHandler(file_handler)
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '100MB' to bytes"""
        units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
        size_str = size_str.upper()
        for unit, multiplier in units.items():
            if size_str.endswith(unit):
                return int(size_str[:-len(unit)]) * multiplier
        return 100 * 1024 * 1024  # Default 100MB
    
    def _get_formatter(self):
        """Get appropriate formatter based on config"""
        if config.log_format == "json":
            return JsonFormatter()
        else:
            return logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def _should_rate_limit(self, error_key: str) -> bool:
        """Check if error should be rate limited"""
        current_time = time.time()
        if current_time - self.last_error_time[error_key] < 60:  # 1 minute window
            self.error_counter[error_key] += 1
            if self.error_counter[error_key] > config.error_rate_limit:
                return True
        else:
            self.error_counter[error_key] = 1
            self.last_error_time[error_key] = current_time
        return False
    
    def _build_context(self, extra: Dict[str, Any] = None) -> Dict[str, Any]:
        """Build logging context with all relevant information"""
        context = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": config.app_name,
            "environment": config.environment,
            "version": config.app_version,
            "instance_id": config.service_instance_id,
            "correlation_id": correlation_id_context.get(),
            "trace_id": trace_id_context.get(),
            "span_id": span_id_context.get()
        }
        
        # Add request context if available
        req_ctx = request_context.get()
        if req_ctx:
            context.update(req_ctx)
        
        # Add system metrics if enabled
        if config.enable_memory_logging:
            process = psutil.Process()
            context["memory_mb"] = process.memory_info().rss / 1024 / 1024
        
        if config.enable_cpu_logging:
            context["cpu_percent"] = psutil.cpu_percent(interval=0)
        
        # Add extra context
        if extra:
            context.update(self._mask_sensitive_data(extra))
        
        # Add custom context fields from config
        if config.custom_context_fields:
            context.update(config.custom_context_fields)
        
        return context
    
    def _mask_sensitive_data(self, data: Any) -> Any:
        """Recursively mask sensitive data"""
        if not config.mask_sensitive_data:
            return data
        
        if isinstance(data, dict):
            return {
                k: config.get_masked_value(v) if config.should_mask_field(k) else self._mask_sensitive_data(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._mask_sensitive_data(item) for item in data]
        elif isinstance(data, str) and len(data) > 100:
            return data[:100] + "...[truncated]"
        return data
    
    def log(self, level: str, message: str, **kwargs):
        """General log method with context"""
        context = self._build_context(kwargs)
        
        if config.log_format == "json":
            log_entry = {
                "level": level.upper(),
                "message": message,
                **context
            }
            getattr(self.logger, level.lower())(json.dumps(log_entry, default=str))
        else:
            getattr(self.logger, level.lower())(message, extra=context)
    
    def debug(self, message: str, **kwargs):
        self.log("debug", message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self.log("info", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.log("warning", message, **kwargs)
    
    def error(self, message: str, error: Exception = None, **kwargs):
        if error and config.capture_stack_traces:
            kwargs["stack_trace"] = traceback.format_exc(limit=config.max_stack_trace_depth)
            kwargs["error_type"] = type(error).__name__
            kwargs["error_message"] = str(error)
        
        # Rate limiting for errors
        error_key = f"{type(error).__name__ if error else 'general'}:{message[:50]}"
        if not self._should_rate_limit(error_key):
            self.log("error", message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self.log("critical", message, **kwargs)


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'created', 'filename', 
                              'funcName', 'levelname', 'levelno', 'lineno', 
                              'module', 'msecs', 'message', 'pathname', 'process',
                              'processName', 'relativeCreated', 'thread', 'threadName']:
                    log_obj[key] = value
        
        return json.dumps(log_obj, default=str)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Main middleware for request/response logging and error handling"""
    
    def __init__(self, app, logger: StructuredLogger):
        super().__init__(app)
        self.logger = logger
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Skip excluded paths
        if not config.should_log_path(request.url.path):
            return await call_next(request)
        
        # Generate or extract correlation ID
        correlation_id = request.headers.get(config.correlation_id_header, str(uuid.uuid4()))
        correlation_id_context.set(correlation_id)
        
        # Set trace and span IDs if distributed tracing is enabled
        if config.enable_distributed_tracing:
            trace_id = request.headers.get(config.trace_id_header, str(uuid.uuid4()))
            span_id = str(uuid.uuid4())
            trace_id_context.set(trace_id)
            span_id_context.set(span_id)
        
        # Capture request details
        start_time = time.time()
        request_body = None
        
        # Store request context
        req_context = {
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_host": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
        }
        
        # Capture request body if enabled
        if config.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body_bytes = await request.body()
                if len(body_bytes) <= config.max_body_log_size:
                    try:
                        request_body = json.loads(body_bytes)
                        req_context["request_body"] = self.logger._mask_sensitive_data(request_body)
                    except json.JSONDecodeError:
                        req_context["request_body"] = body_bytes.decode('utf-8', errors='ignore')[:1000]
                else:
                    req_context["request_body"] = f"[Body too large: {len(body_bytes)} bytes]"
                
                # Reconstruct request for downstream
                async def receive():
                    return {"type": "http.request", "body": body_bytes}
                request = Request(request.scope, receive, request._send)
            except Exception as e:
                self.logger.warning(f"Failed to capture request body: {e}")
        
        # Capture request headers if enabled
        if config.log_request_headers:
            req_context["request_headers"] = self.logger._mask_sensitive_data(dict(request.headers))
        
        request_context.set(req_context)
        
        # Log incoming request
        self.logger.info(
            f"Request started: {request.method} {request.url.path}",
            **req_context
        )
        
        # Process request
        response = None
        error_occurred = False
        
        try:
            response = await call_next(request)
            
            # Capture response body if enabled
            if config.log_response_body and response.status_code != 204:
                response_body = b""
                async for chunk in response.body_iterator:
                    response_body += chunk
                
                try:
                    if len(response_body) <= config.max_body_log_size:
                        response_content = json.loads(response_body)
                        req_context["response_body"] = self.logger._mask_sensitive_data(response_content)
                    else:
                        req_context["response_body"] = f"[Body too large: {len(response_body)} bytes]"
                except:
                    req_context["response_body"] = response_body.decode('utf-8', errors='ignore')[:1000]
                
                # Recreate response
                response = Response(
                    content=response_body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type
                )
            
        except Exception as e:
            error_occurred = True
            self.logger.error(
                f"Unhandled exception during request processing",
                error=e,
                **req_context
            )
            response = JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "correlation_id": correlation_id}
            )
        
        # Calculate performance metrics
        duration_ms = (time.time() - start_time) * 1000
        
        # Determine performance category
        performance_category = "fast"
        for category, threshold in sorted(PERFORMANCE_THRESHOLDS.items(), key=lambda x: x[1], reverse=True):
            if duration_ms >= threshold:
                performance_category = category
                break
        
        # Log response
        log_data = {
            **req_context,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "performance_category": performance_category
        }
        
        # Add response headers if enabled
        if config.log_response_headers:
            log_data["response_headers"] = dict(response.headers)
        
        # Determine log level based on status code and performance
        if response.status_code >= 500 or error_occurred:
            self.logger.error(f"Request failed: {request.method} {request.url.path}", **log_data)
        elif response.status_code >= 400:
            self.logger.warning(f"Client error: {request.method} {request.url.path}", **log_data)
        elif duration_ms > config.slow_request_threshold_ms:
            self.logger.warning(f"Slow request detected: {request.method} {request.url.path}", **log_data)
        else:
            self.logger.info(f"Request completed: {request.method} {request.url.path}", **log_data)
        
        # Add correlation ID to response headers
        response.headers[config.correlation_id_header] = correlation_id
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        
        return response


class ErrorHandlerRegistry:
    """Registry for custom exception handlers with advanced features"""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.handlers: Dict[type, Callable] = {}
        self.error_stats = defaultdict(lambda: {"count": 0, "last_seen": None})
    
    def register(self, exception_class: type, handler: Callable):
        """Register a custom exception handler"""
        self.handlers[exception_class] = handler
    
    def handle(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle exception with appropriate handler"""
        correlation_id = correlation_id_context.get()
        
        # Update error statistics
        error_key = f"{type(exc).__name__}"
        self.error_stats[error_key]["count"] += 1
        self.error_stats[error_key]["last_seen"] = datetime.now(timezone.utc).isoformat()
        
        # Find and execute handler
        for exc_type, handler in self.handlers.items():
            if isinstance(exc, exc_type):
                return handler(request, exc, correlation_id)
        
        # Default handler
        self.logger.error(
            f"Unhandled exception: {type(exc).__name__}",
            error=exc,
            path=request.url.path,
            method=request.method
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    def get_stats(self) -> Dict:
        """Get error statistics"""
        return dict(self.error_stats)


def setup_observability(app: FastAPI) -> FastAPI:
    """
    Main setup function to configure all observability features.
    Call this in your main.py: app = setup_observability(app)
    """
    
    # Initialize logger
    logger = StructuredLogger(config.app_name)
    
    # Initialize error handler registry
    error_registry = ErrorHandlerRegistry(logger)
    
    # Add middleware
    app.add_middleware(ObservabilityMiddleware, logger=logger)
    
    # Register exception handlers
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        correlation_id = correlation_id_context.get()
        
        logger.warning(
            f"HTTP exception: {exc.status_code}",
            status_code=exc.status_code,
            detail=exc.detail,
            path=request.url.path,
            method=request.method
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        correlation_id = correlation_id_context.get()
        
        logger.warning(
            "Validation error",
            errors=exc.errors(),
            body=exc.body,
            path=request.url.path,
            method=request.method
        )
        
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation failed",
                "details": exc.errors(),
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        return error_registry.handle(request, exc)
    
    # Add health check endpoint if enabled
    if config.enable_health_check:
        @app.get(config.health_check_path, tags=["monitoring"])
        async def health_check():
            """Health check endpoint with deep checks if enabled"""
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "service": config.app_name,
                "version": config.app_version,
                "environment": config.environment
            }
            
            if config.deep_health_check:
                try:
                    # Add system metrics
                    process = psutil.Process()
                    health_status["metrics"] = {
                        "memory_mb": process.memory_info().rss / 1024 / 1024,
                        "cpu_percent": process.cpu_percent(interval=0),
                        "open_files": len(process.open_files()),
                        "num_threads": process.num_threads(),
                        "uptime_seconds": time.time() - process.create_time()
                    }
                    
                    # Add error statistics
                    health_status["error_stats"] = error_registry.get_stats()
                    
                except Exception as e:
                    health_status["status"] = "degraded"
                    health_status["error"] = str(e)
            
            return health_status
    
    # Add metrics endpoint if enabled
    if config.enable_metrics:
        @app.get(config.metrics_endpoint, tags=["monitoring"])
        async def metrics():
            """Prometheus-compatible metrics endpoint"""
            metrics_data = []
            
            # Add custom metrics
            process = psutil.Process()
            metrics_data.append(f"# HELP {config.custom_metrics_prefix}_memory_bytes Process memory usage")
            metrics_data.append(f"# TYPE {config.custom_metrics_prefix}_memory_bytes gauge")
            metrics_data.append(f"{config.custom_metrics_prefix}_memory_bytes {process.memory_info().rss}")
            
            metrics_data.append(f"# HELP {config.custom_metrics_prefix}_cpu_percent Process CPU usage")
            metrics_data.append(f"# TYPE {config.custom_metrics_prefix}_cpu_percent gauge")
            metrics_data.append(f"{config.custom_metrics_prefix}_cpu_percent {process.cpu_percent(interval=0)}")
            
            # Add error metrics
            error_stats = error_registry.get_stats()
            for error_type, stats in error_stats.items():
                safe_error_type = error_type.replace(" ", "_").lower()
                metrics_data.append(f"# HELP {config.custom_metrics_prefix}_errors_total Total errors by type")
                metrics_data.append(f"# TYPE {config.custom_metrics_prefix}_errors_total counter")
                metrics_data.append(f'{config.custom_metrics_prefix}_errors_total{{type="{safe_error_type}"}} {stats["count"]}')
            
            return Response(content="\n".join(metrics_data), media_type="text/plain")
    
    # Store references in app state
    app.state.logger = logger
    app.state.error_registry = error_registry
    
    logger.info(
        f"Observability initialized for {config.app_name}",
        environment=config.environment,
        version=config.app_version,
        features={
            "correlation_id": config.enable_correlation_id,
            "distributed_tracing": config.enable_distributed_tracing,
            "metrics": config.enable_metrics,
            "health_check": config.enable_health_check,
            "performance_logging": config.enable_performance_logging
        }
    )
    
    return app


# Decorator for custom function logging
def log_function(level: str = "info", include_args: bool = True, include_result: bool = False):
    """Decorator to log function calls with context"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = StructuredLogger(f"{config.app_name}.{func.__module__}")
            
            log_data = {
                "function": func.__name__,
                "module": func.__module__
            }
            
            if include_args:
                log_data["args"] = str(args)[:500] if args else None
                log_data["kwargs"] = logger._mask_sensitive_data(kwargs) if kwargs else None
            
            logger.log(level, f"Function called: {func.__name__}", **log_data)
            
            try:
                result = await func(*args, **kwargs)
                
                if include_result:
                    log_data["result"] = str(result)[:500]
                
                logger.log(level, f"Function completed: {func.__name__}", **log_data)
                return result
                
            except Exception as e:
                logger.error(f"Function failed: {func.__name__}", error=e, **log_data)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = StructuredLogger(f"{config.app_name}.{func.__module__}")
            
            log_data = {
                "function": func.__name__,
                "module": func.__module__
            }
            
            if include_args:
                log_data["args"] = str(args)[:500] if args else None
                log_data["kwargs"] = logger._mask_sensitive_data(kwargs) if kwargs else None
            
            logger.log(level, f"Function called: {func.__name__}", **log_data)
            
            try:
                result = func(*args, **kwargs)
                
                if include_result:
                    log_data["result"] = str(result)[:500]
                
                logger.log(level, f"Function completed: {func.__name__}", **log_data)
                return result
                
            except Exception as e:
                logger.error(f"Function failed: {func.__name__}", error=e, **log_data)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# Export main components
__all__ = [
    'setup_observability',
    'StructuredLogger',
    'ObservabilityMiddleware',
    'ErrorHandlerRegistry',
    'log_function',
    'correlation_id_context',
    'trace_id_context',
    'span_id_context'
]
