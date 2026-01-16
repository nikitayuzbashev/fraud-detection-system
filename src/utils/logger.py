"""Logging configuration for the fraud detection system."""

import logging
import sys
from typing import Optional
from pythonjsonlogger import jsonlogger
from .config import Config


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ('json' or 'standard')
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Set level from config or parameter
    log_level = level or Config.LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level.upper()))
    
    # Set formatter based on format type
    format_type = log_format or Config.LOG_FORMAT
    
    if format_type.lower() == "json":
        # JSON formatter for production/structured logging
        formatter = jsonlogger.JsonFormatter(
            "%(timestamp)s %(name)s %(levelname)s %(message)s",
            rename_fields={"levelname": "severity", "name": "logger"},
            datefmt="%Y-%m-%dT%H:%M:%S"
        )
    else:
        # Standard formatter for development
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


# Create default logger
logger = setup_logger(__name__)
