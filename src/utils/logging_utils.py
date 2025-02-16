"""
DrugFormDB Logging Utilities
------------------------

This module provides advanced logging configuration and handlers
for the DrugFormDB project.

Author: Ahmad Rufai Yusuf
License: MIT
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union, Dict
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
from datetime import datetime

from .config import LOGGING_CONFIG, get_project_root

class JsonFormatter(logging.Formatter):
    """
    Custom formatter that outputs log records as JSON.
    """
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        # Basic log data
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": str(record.exc_info[0].__name__),
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields if present
        if hasattr(record, "extra"):
            log_data["extra"] = record.extra
        
        return json.dumps(log_data)

def setup_logger(
    name: str,
    level: Union[int, str] = LOGGING_CONFIG["level"],
    log_file: Optional[Union[str, Path]] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    format_json: bool = False,
    console: bool = True,
    propagate: bool = False
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Path to log file
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        format_json: Whether to format logs as JSON
        console: Whether to log to console
        propagate: Whether to propagate to parent loggers
    
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create formatter
    if format_json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt=LOGGING_CONFIG["format"],
            datefmt=LOGGING_CONFIG["date_format"]
        )
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if log file specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def setup_daily_logger(
    name: str,
    log_dir: Optional[Union[str, Path]] = None,
    level: Union[int, str] = LOGGING_CONFIG["level"],
    backup_count: int = 30,
    format_json: bool = False,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger that rotates daily.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        backup_count: Number of backup files to keep
        format_json: Whether to format logs as JSON
        console: Whether to log to console
    
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create formatter
    if format_json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt=LOGGING_CONFIG["format"],
            datefmt=LOGGING_CONFIG["date_format"]
        )
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add daily rotating file handler if log directory specified
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{name}.log"
        file_handler = TimedRotatingFileHandler(
            str(log_file),
            when="midnight",
            interval=1,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class LoggerAdapter(logging.LoggerAdapter):
    """
    Custom logger adapter that adds context to log messages.
    """
    def __init__(
        self,
        logger: logging.Logger,
        extra: Optional[Dict] = None
    ):
        """
        Initialize the adapter with a logger and extra context.
        
        Args:
            logger: Base logger
            extra: Extra context to add to all messages
        """
        super().__init__(logger, extra or {})
    
    def process(
        self,
        msg: str,
        kwargs: Dict
    ) -> tuple[str, Dict]:
        """
        Process the log message and kwargs.
        
        Args:
            msg: Log message
            kwargs: Keyword arguments
        
        Returns:
            Tuple of (modified message, modified kwargs)
        """
        # Add context to message if extra data exists
        if self.extra:
            context_str = " ".join(
                f"[{k}={v}]" for k, v in self.extra.items()
            )
            msg = f"{context_str} {msg}"
        
        return msg, kwargs

def get_logger(
    name: str,
    extra: Optional[Dict] = None
) -> Union[logging.Logger, LoggerAdapter]:
    """
    Get a logger with optional context.
    
    Args:
        name: Logger name
        extra: Extra context to add to all messages
    
    Returns:
        Union[logging.Logger, LoggerAdapter]: Logger or adapter
    """
    logger = logging.getLogger(name)
    
    if extra:
        return LoggerAdapter(logger, extra)
    return logger

# Example usage
if __name__ == "__main__":
    # Set up a basic logger
    logger = setup_logger(
        "example",
        log_file=get_project_root() / "logs" / "example.log",
        format_json=True
    )
    
    # Log some messages
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.error("An error occurred", exc_info=True)
    
    # Set up a daily logger
    daily_logger = setup_daily_logger(
        "daily_example",
        log_dir=get_project_root() / "logs",
        format_json=True
    )
    
    # Log with context
    context_logger = get_logger(
        "context_example",
        extra={"user": "test_user", "session": "123"}
    )
    context_logger.info("This is a message with context") 