"""
Structured logging setup for drone racing system.

Provides consistent, informative logging across all modules with
support for file output and colored console output.

Usage:
    from utils.logger import setup_logger, get_logger

    # Initialize once at startup
    setup_logger()

    # Get logger in any module
    logger = get_logger(__name__)
    logger.info("Taking off", altitude=5.0)
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog
from structlog.typing import Processor

from config.settings import get_settings


def setup_logger(
    log_level: Optional[str] = None,
    log_to_file: Optional[bool] = None,
    log_directory: Optional[str] = None,
) -> None:
    """
    Initialize the logging system.

    Should be called once at application startup before any logging occurs.
    Uses settings from config if parameters are not explicitly provided.

    Args:
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Override whether to log to file
        log_directory: Override directory for log files
    """
    settings = get_settings()

    # Use provided values or fall back to settings
    level = log_level or settings.logging.log_level
    to_file = log_to_file if log_to_file is not None else settings.logging.log_to_file
    log_dir = log_directory or settings.logging.log_directory

    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        level=numeric_level,
        stream=sys.stdout,
    )

    # Build processor chain for structlog
    processors: list[Processor] = [
        # Add log level to event dict
        structlog.stdlib.add_log_level,
        # Add logger name
        structlog.stdlib.add_logger_name,
        # Add timestamp
        structlog.processors.TimeStamper(fmt="iso"),
        # Handle exceptions
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        # Add call site information in debug mode
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ) if numeric_level == logging.DEBUG else structlog.stdlib.ProcessorFormatter.remove_processors_meta,
    ]

    # Add console renderer (colored for terminal)
    if sys.stdout.isatty():
        # Colored output for terminal
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            )
        )
    else:
        # JSON output for non-terminal (useful for log aggregation)
        processors.append(structlog.processors.JSONRenderer())

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set up file logging if enabled
    if to_file:
        _setup_file_logging(log_dir, numeric_level)


def _setup_file_logging(log_directory: str, level: int) -> None:
    """
    Set up file-based logging.

    Creates a timestamped log file in the specified directory.

    Args:
        log_directory: Directory to store log files
        level: Logging level
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_directory)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"drone_racing_{timestamp}.log"

    # Add file handler to root logger
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Use a simpler format for file logging
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    logging.getLogger().addHandler(file_handler)

    # Log the file location
    print(f"Logging to file: {log_file}")


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance for the given module name.

    Args:
        name: Module name (typically __name__)

    Returns:
        BoundLogger instance for structured logging

    Example:
        logger = get_logger(__name__)
        logger.info("Connected to drone", system_id=1)
        logger.warning("Low battery", voltage=10.5, threshold=11.0)
        logger.error("Connection lost", error="timeout")
    """
    return structlog.get_logger(name)


# Convenience function for quick logging without setup
def log_info(message: str, **kwargs) -> None:
    """Quick info logging without getting a logger instance."""
    logger = get_logger("drone_racing")
    logger.info(message, **kwargs)


def log_warning(message: str, **kwargs) -> None:
    """Quick warning logging without getting a logger instance."""
    logger = get_logger("drone_racing")
    logger.warning(message, **kwargs)


def log_error(message: str, **kwargs) -> None:
    """Quick error logging without getting a logger instance."""
    logger = get_logger("drone_racing")
    logger.error(message, **kwargs)


def log_debug(message: str, **kwargs) -> None:
    """Quick debug logging without getting a logger instance."""
    logger = get_logger("drone_racing")
    logger.debug(message, **kwargs)
