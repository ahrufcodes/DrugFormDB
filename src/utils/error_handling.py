"""
DrugFormDB Error Handling Module
----------------------------

This module provides custom exceptions and error handling utilities
for the DrugFormDB project.

Author: Ahmad Rufai Yusuf
License: MIT
"""

import logging
from typing import Optional, Any, Callable
from functools import wraps
import traceback
import time

# Configure logging
logger = logging.getLogger(__name__)

class DrugFormDBError(Exception):
    """Base exception class for DrugFormDB."""
    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class ValidationError(DrugFormDBError):
    """Raised when validation fails."""
    pass

class ClassificationError(DrugFormDBError):
    """Raised when drug classification fails."""
    pass

class APIError(DrugFormDBError):
    """Raised when API calls fail."""
    pass

class DataProcessingError(DrugFormDBError):
    """Raised when data processing operations fail."""
    pass

class ConfigurationError(DrugFormDBError):
    """Raised when configuration is invalid."""
    pass

def retry_on_exception(
    retries: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,),
    logger: Optional[logging.Logger] = None
) -> Callable:
    """
    Decorator to retry a function on exception.
    
    Args:
        retries: Number of retries
        delay: Delay between retries in seconds
        exceptions: Tuple of exceptions to catch
        logger: Logger instance
    
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if logger:
                        logger.warning(
                            f"Attempt {attempt + 1}/{retries} failed: {str(e)}"
                        )
                    if attempt < retries - 1:
                        time.sleep(delay)
            
            if logger:
                logger.error(
                    f"All {retries} attempts failed. "
                    f"Last error: {str(last_exception)}"
                )
            raise last_exception
        
        return wrapper
    return decorator

def log_exceptions(
    logger: Optional[logging.Logger] = None,
    level: int = logging.ERROR,
    include_traceback: bool = True
) -> Callable:
    """
    Decorator to log exceptions.
    
    Args:
        logger: Logger instance
        level: Logging level
        include_traceback: Whether to include traceback
    
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log = logger or logging.getLogger(__name__)
                error_msg = f"Error in {func.__name__}: {str(e)}"
                
                if include_traceback:
                    error_msg += f"\nTraceback:\n{traceback.format_exc()}"
                
                log.log(level, error_msg)
                raise
        
        return wrapper
    return decorator

def handle_api_error(error: Exception) -> APIError:
    """
    Convert API exceptions to APIError.
    
    Args:
        error: Original exception
    
    Returns:
        APIError: Converted error
    """
    error_msg = str(error)
    error_code = getattr(error, 'code', None)
    
    if 'rate limit' in error_msg.lower():
        return APIError(
            "API rate limit exceeded. Please try again later.",
            error_code="RATE_LIMIT"
        )
    elif 'timeout' in error_msg.lower():
        return APIError(
            "API request timed out. Please try again.",
            error_code="TIMEOUT"
        )
    elif 'unauthorized' in error_msg.lower():
        return APIError(
            "API authentication failed. Please check your credentials.",
            error_code="AUTH_ERROR"
        )
    else:
        return APIError(
            f"API error occurred: {error_msg}",
            error_code="API_ERROR"
        )

def safe_execute(
    func: Callable,
    *args: Any,
    default_value: Any = None,
    **kwargs: Any
) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        default_value: Value to return on error
        **kwargs: Keyword arguments
    
    Returns:
        Any: Function result or default value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error executing {func.__name__}: {str(e)}")
        return default_value

# Example usage
if __name__ == "__main__":
    # Configure test logger
    logging.basicConfig(level=logging.INFO)
    test_logger = logging.getLogger(__name__)
    
    # Example function with retry
    @retry_on_exception(retries=3, delay=1, logger=test_logger)
    def example_retry_function():
        raise ValueError("Test error")
    
    # Example function with exception logging
    @log_exceptions(logger=test_logger)
    def example_logging_function():
        raise RuntimeError("Test error")
    
    try:
        example_retry_function()
    except ValueError:
        logger.info("Retry example completed")
    
    try:
        example_logging_function()
    except RuntimeError:
        logger.info("Logging example completed") 