import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_dir: str = "logs", 
                 enable_file_logging: bool = True, enable_console: bool = True):
    """
    Setup comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        enable_file_logging: Whether to enable file logging
        enable_console: Whether to enable console logging
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Formatter with more details
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Simple formatter for console
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
    
    if enable_file_logging:
        # General log file with rotation
        general_log_file = log_path / "task_runner.log"
        file_handler = logging.handlers.RotatingFileHandler(
            general_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        # Error-only log file
        error_log_file = log_path / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
        
        # Performance log (for LLM calls and processing times)
        perf_log_file = log_path / "performance.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=2
        )
        perf_handler.setLevel(logging.INFO)
        perf_formatter = logging.Formatter(
            '%(asctime)s - PERF - %(message)s'
        )
        perf_handler.setFormatter(perf_formatter)
        
        # Create performance logger
        perf_logger = logging.getLogger('performance')
        perf_logger.addHandler(perf_handler)
        perf_logger.propagate = False  # Don't propagate to root logger
    
    # Configure specific loggers
    
    # Suppress noisy external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('tinydb').setLevel(logging.WARNING)
    
    # Set task runner components to appropriate levels
    logging.getLogger('task_processor').setLevel(logging.INFO)
    logging.getLogger('llm_processor').setLevel(logging.INFO)
    logging.getLogger('database_manager').setLevel(logging.INFO)
    logging.getLogger('file_validator').setLevel(logging.INFO)

def get_performance_logger():
    """Get the performance logger instance."""
    return logging.getLogger('performance')

def log_performance(operation: str, duration: float, additional_info: dict = None):
    """
    Log performance metrics.
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
        additional_info: Additional information to log
    """
    perf_logger = get_performance_logger()
    
    info_str = f"Operation: {operation}, Duration: {duration:.3f}s"
    if additional_info:
        info_parts = [f"{k}: {v}" for k, v in additional_info.items()]
        info_str += f", {', '.join(info_parts)}"
    
    perf_logger.info(info_str)

class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str, additional_info: dict = None):
        self.operation_name = operation_name
        self.additional_info = additional_info or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            
            # Add exception info if operation failed
            if exc_type:
                self.additional_info['error'] = str(exc_val)
                self.additional_info['exception_type'] = exc_type.__name__
            
            log_performance(self.operation_name, duration, self.additional_info)

def setup_exception_logging():
    """Setup global exception handling."""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log KeyboardInterrupt
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger = logging.getLogger(__name__)
        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    import sys
    sys.excepthook = handle_exception

# Custom filters for sensitive data
class SensitiveDataFilter(logging.Filter):
    """Filter out sensitive information from logs."""
    
    SENSITIVE_PATTERNS = [
        r'api[_-]?key["\']?\s*[:=]\s*["\']?([^"\'\s]+)',
        r'password["\']?\s*[:=]\s*["\']?([^"\'\s]+)',
        r'token["\']?\s*[:=]\s*["\']?([^"\'\s]+)',
        r'secret["\']?\s*[:=]\s*["\']?([^"\'\s]+)',
    ]
    
    def filter(self, record):
        """Filter sensitive data from log records."""
        import re
        
        # Check message for sensitive patterns
        message = record.getMessage()
        
        for pattern in self.SENSITIVE_PATTERNS:
            message = re.sub(pattern, r'\1=***REDACTED***', message, flags=re.IGNORECASE)
        
        # Replace the record's message
        record.msg = message
        record.args = ()
        
        return True

def add_sensitive_data_filter():
    """Add sensitive data filter to all handlers."""
    sensitive_filter = SensitiveDataFilter()
    
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.addFilter(sensitive_filter)