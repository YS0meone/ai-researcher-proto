"""
Logging configuration for the AI Researcher application.

Usage:
    from app.core.logging_config import setup_logging

    # In your main app or module entry point
    setup_logging()

    # Then use standard logging
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Hello world")
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from app.core.config import settings


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',       # Reset
    }

    def format(self, record):
        # Add color to level name
        if sys.stderr.isatty():  # Only use colors if output is a terminal
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"

        return super().format(record)


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to settings.LOG_LEVEL.
        log_file: Path to log file. Defaults to logs/app.log
        console_output: Whether to output logs to console (default: True)
        file_output: Whether to output logs to file (default: True)
    """
    # Get log level from settings or parameter
    log_level = level or getattr(settings, 'LOG_LEVEL', 'INFO')
    log_level = getattr(logging, log_level.upper())

    # Create logs directory if it doesn't exist
    if file_output:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_file or str(log_dir / "app.log")

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler with colored output
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        # Use colored formatter for console
        console_format = '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s'
        console_formatter = ColoredFormatter(
            console_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if file_output and log_file:
        # Rotating file handler (max 10MB per file, keep 5 backup files)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)

        # More detailed format for file output (includes module and function)
        file_format = '%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s'
        file_formatter = logging.Formatter(
            file_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Reduce verbosity of some noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)

    # Log initialization
    root_logger.info("=" * 80)
    root_logger.info(f"Logging initialized - Level: {logging.getLevelName(log_level)}")
    if file_output and log_file:
        root_logger.info(f"Log file: {log_file}")
    root_logger.info("=" * 80)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Hello world")
    """
    return logging.getLogger(name)
