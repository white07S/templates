"""
Advanced Configuration Module for FastAPI Observability
"""
import os
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseSettings, Field, validator
from datetime import timedelta


class LogLevel(str, Enum):
    """Log levels enum"""
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    TRACE = "TRACE"


class Environment(str, Enum):
    """Application environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogFormat(str, Enum):
    """Log output formats"""
    JSON = "json"
    PRETTY = "pretty"
    COMPACT = "compact"


class ObservabilityConfig(BaseSettings):
    """
    Centralized observability configuration using Pydantic BaseSettings.
    Loads from environment variables with OBS_ prefix.
    """
    
    # Core Settings
    app_name: str = Field("fastapi-app", env="OBS_APP_NAME")
    app_version: str = Field("1.0.0", env="OBS_APP_VERSION")
    environment: Environment = Field(Environment.DEVELOPMENT, env="OBS_ENVIRONMENT")
    service_instance_id: str = Field("", env="OBS_SERVICE_INSTANCE_ID")
    
    # Logging Configuration
    log_level: LogLevel = Field(LogLevel.INFO, env="OBS_LOG_LEVEL")
    log_format: LogFormat = Field(LogFormat.JSON, env="OBS_LOG_FORMAT")
    log_file_path: Optional[str] = Field(None, env="OBS_LOG_FILE_PATH")
    log_to_console: bool = Field(True, env="OBS_LOG_TO_CONSOLE")
    log_to_file: bool = Field(False, env="OBS_LOG_TO_FILE")
    log_rotation_size: str = Field("100MB", env="OBS_LOG_ROTATION_SIZE")
    log_retention_days: int = Field(30, env="OBS_LOG_RETENTION_DAYS")
    
    # Request/Response Logging
    log_request_body: bool = Field(True, env="OBS_LOG_REQUEST_BODY")
    log_response_body: bool = Field(True, env="OBS_LOG_RESPONSE_BODY")
    log_request_headers: bool = Field(True, env="OBS_LOG_REQUEST_HEADERS")
    log_response_headers: bool = Field(False, env="OBS_LOG_RESPONSE_HEADERS")
    max_body_log_size: int = Field(10000, env="OBS_MAX_BODY_LOG_SIZE")  # bytes
    
    # Sensitive Data Masking
    mask_sensitive_data: bool = Field(True, env="OBS_MASK_SENSITIVE_DATA")
    sensitive_fields: List[str] = Field(
        default_factory=lambda: [
            "password", "token", "api_key", "secret", "authorization",
            "cookie", "session", "credit_card", "ssn", "pin", "cvv"
        ],
        env="OBS_SENSITIVE_FIELDS"
    )
    mask_pattern: str = Field("****", env="OBS_MASK_PATTERN")
    
    # Performance Monitoring
    enable_performance_logging: bool = Field(True, env="OBS_ENABLE_PERFORMANCE_LOGGING")
    slow_request_threshold_ms: int = Field(1000, env="OBS_SLOW_REQUEST_THRESHOLD_MS")
    enable_memory_logging: bool = Field(True, env="OBS_ENABLE_MEMORY_LOGGING")
    enable_cpu_logging: bool = Field(True, env="OBS_ENABLE_CPU_LOGGING")
    
    # Error Tracking
    capture_stack_traces: bool = Field(True, env="OBS_CAPTURE_STACK_TRACES")
    max_stack_trace_depth: int = Field(10, env="OBS_MAX_STACK_TRACE_DEPTH")
    group_similar_errors: bool = Field(True, env="OBS_GROUP_SIMILAR_ERRORS")
    error_rate_limit: int = Field(100, env="OBS_ERROR_RATE_LIMIT")  # per minute
    
    # Correlation and Tracing
    enable_correlation_id: bool = Field(True, env="OBS_ENABLE_CORRELATION_ID")
    correlation_id_header: str = Field("X-Correlation-ID", env="OBS_CORRELATION_ID_HEADER")
    enable_distributed_tracing: bool = Field(True, env="OBS_ENABLE_DISTRIBUTED_TRACING")
    trace_id_header: str = Field("X-Trace-ID", env="OBS_TRACE_ID_HEADER")
    span_id_header: str = Field("X-Span-ID", env="OBS_SPAN_ID_HEADER")
    
    # Metrics Collection
    enable_metrics: bool = Field(True, env="OBS_ENABLE_METRICS")
    metrics_endpoint: str = Field("/metrics", env="OBS_METRICS_ENDPOINT")
    custom_metrics_prefix: str = Field("fastapi", env="OBS_CUSTOM_METRICS_PREFIX")
    
    # Health Check
    enable_health_check: bool = Field(True, env="OBS_ENABLE_HEALTH_CHECK")
    health_check_path: str = Field("/health", env="OBS_HEALTH_CHECK_PATH")
    deep_health_check: bool = Field(True, env="OBS_DEEP_HEALTH_CHECK")
    
    # Rate Limiting for Logs
    enable_log_sampling: bool = Field(False, env="OBS_ENABLE_LOG_SAMPLING")
    log_sample_rate: float = Field(1.0, env="OBS_LOG_SAMPLE_RATE")  # 1.0 = 100%
    
    # Alerting Configuration
    enable_alerts: bool = Field(False, env="OBS_ENABLE_ALERTS")
    alert_webhook_url: Optional[str] = Field(None, env="OBS_ALERT_WEBHOOK_URL")
    alert_threshold_error_rate: float = Field(0.05, env="OBS_ALERT_THRESHOLD_ERROR_RATE")
    alert_threshold_response_time: int = Field(5000, env="OBS_ALERT_THRESHOLD_RESPONSE_TIME")
    
    # External Integrations
    enable_sentry: bool = Field(False, env="OBS_ENABLE_SENTRY")
    sentry_dsn: Optional[str] = Field(None, env="OBS_SENTRY_DSN")
    enable_datadog: bool = Field(False, env="OBS_ENABLE_DATADOG")
    datadog_api_key: Optional[str] = Field(None, env="OBS_DATADOG_API_KEY")
    
    # Excluded Paths (no logging)
    excluded_paths: List[str] = Field(
        default_factory=lambda: [
            "/health", "/metrics", "/docs", "/redoc", 
            "/openapi.json", "/favicon.ico"
        ],
        env="OBS_EXCLUDED_PATHS"
    )
    
    # Custom Context Fields
    custom_context_fields: Dict[str, Any] = Field(
        default_factory=dict,
        env="OBS_CUSTOM_CONTEXT_FIELDS"
    )
    
    class Config:
        env_prefix = "OBS_"
        env_file = ".env"
        case_sensitive = False
        use_enum_values = True
    
    @validator("service_instance_id", pre=True, always=True)
    def generate_instance_id(cls, v):
        """Generate instance ID if not provided"""
        if not v:
            import socket
            import uuid
            return f"{socket.gethostname()}-{str(uuid.uuid4())[:8]}"
        return v
    
    @validator("log_sample_rate")
    def validate_sample_rate(cls, v):
        """Ensure sample rate is between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError("log_sample_rate must be between 0 and 1")
        return v
    
    def get_log_level_int(self) -> int:
        """Convert log level to integer for comparison"""
        levels = {
            LogLevel.TRACE: 5,
            LogLevel.DEBUG: 10,
            LogLevel.INFO: 20,
            LogLevel.WARNING: 30,
            LogLevel.ERROR: 40,
            LogLevel.CRITICAL: 50
        }
        return levels.get(self.log_level, 20)
    
    def should_log_path(self, path: str) -> bool:
        """Check if path should be logged"""
        return not any(path.startswith(excluded) for excluded in self.excluded_paths)
    
    def should_mask_field(self, field_name: str) -> bool:
        """Check if field should be masked"""
        if not self.mask_sensitive_data:
            return False
        field_lower = field_name.lower()
        return any(sensitive in field_lower for sensitive in self.sensitive_fields)
    
    def get_masked_value(self, value: Any) -> Any:
        """Get masked version of sensitive value"""
        if value is None:
            return None
        if isinstance(value, str):
            if len(value) <= 4:
                return self.mask_pattern
            return f"{value[:2]}{self.mask_pattern}{value[-2:]}"
        return self.mask_pattern


# Singleton instance
config = ObservabilityConfig()


# Logging format templates
LOG_FORMAT_TEMPLATES = {
    LogFormat.JSON: {
        "format": "json",
        "fields": [
            "timestamp", "level", "correlation_id", "trace_id",
            "service", "environment", "message", "context"
        ]
    },
    LogFormat.PRETTY: {
        "format": "%(asctime)s | %(levelname)-8s | %(correlation_id)s | %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S"
    },
    LogFormat.COMPACT: {
        "format": "%(levelname)s:%(name)s:%(message)s"
    }
}


# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "fast": 100,      # ms
    "normal": 500,    # ms
    "slow": 1000,     # ms
    "critical": 5000  # ms
}


# HTTP Status Code Categories
STATUS_CODE_CATEGORIES = {
    "success": range(200, 300),
    "redirect": range(300, 400),
    "client_error": range(400, 500),
    "server_error": range(500, 600)
}
