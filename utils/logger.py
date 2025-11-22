"""
Central Logging System for NES-CARLA Project
===========================================

Provides centralized, configurable logging across all project components.
Supports different log levels for various types of information:

- DEBUG: Detailed predictions, model outputs, internal states
- INFO: Reward calculations, route progress, general status
- WARNING: Penalties, safety violations, performance issues  
- ERROR: Critical failures, exceptions
- CRITICAL: System-breaking errors

Usage:
    from utils.logger import get_logger
    
    logger = get_logger(__name__)
    logger.info("Agent reached waypoint")
    logger.warning("Collision penalty applied")
    logger.debug(f"Model prediction: {prediction}")

Configuration:
    Set logging level via LOG_LEVEL environment variable or modify DEFAULT_LEVEL
    Example: export LOG_LEVEL=INFO
"""

import logging
import os
import sys
from typing import Optional
from datetime import datetime

# Default logging configuration
DEFAULT_LEVEL = "ERROR"  # Set to ERROR by default
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"

# Color codes for terminal output
class LogColors:
    """ANSI color codes for colored terminal output"""
    DEBUG = '\033[36m'      # Cyan
    INFO = '\033[32m'       # Green  
    WARNING = '\033[33m'    # Yellow
    ERROR = '\033[31m'      # Red
    CRITICAL = '\033[35m'   # Magenta
    RESET = '\033[0m'       # Reset color

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels in terminal output"""
    
    COLORS = {
        logging.DEBUG: LogColors.DEBUG,
        logging.INFO: LogColors.INFO,
        logging.WARNING: LogColors.WARNING,
        logging.ERROR: LogColors.ERROR,
        logging.CRITICAL: LogColors.CRITICAL,
    }
    
    def format(self, record):
        # Get the original formatted message
        log_message = super().format(record)
        
        # Add color if outputting to terminal
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            color = self.COLORS.get(record.levelno, LogColors.RESET)
            return f"{color}{log_message}{LogColors.RESET}"
        
        return log_message

# Global logger configuration
_loggers = {}
_configured = False

def configure_logging(level: Optional[str] = None, 
                     log_file: Optional[str] = None,
                     enable_colors: bool = True,
                     force_reconfigure: bool = False) -> None:
    """
    Configure the global logging system for the project.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        enable_colors: Whether to use colored output in terminal
        force_reconfigure: Force reconfiguration even if already configured
    """
    global _configured
    
    if _configured and not force_reconfigure:
        # Allow level change without full reconfiguration
        if level is not None:
            set_log_level(level)
        return
    
    # Determine logging level
    if level is None:
        level = os.getenv('LOG_LEVEL', DEFAULT_LEVEL)
    
    log_level = getattr(logging, level.upper(), logging.WARNING)
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Set formatter (with or without colors)
    if enable_colors:
        console_formatter = ColoredFormatter(LOG_FORMAT, DATE_FORMAT)
    else:
        console_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    _configured = True
    
    # Log configuration info only if level allows it
    if log_level <= logging.INFO:
        root_logger.info(f"🔧 Logging configured - Level: {level.upper()}, Colors: {enable_colors}")
        if log_file:
            root_logger.info(f"📁 Log file: {log_file}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module/component.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Ensure logging is configured
    if not _configured:
        configure_logging()
    
    # Return cached logger or create new one
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    
    return _loggers[name]

def set_log_level(level: str) -> None:
    """
    Change the logging level at runtime.
    
    Args:
        level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_level = getattr(logging, level.upper(), logging.WARNING)
    
    # Update root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Update all handlers
    for handler in root_logger.handlers:
        handler.setLevel(log_level)
    
    # Update all existing named loggers
    for logger_name in _loggers:
        logger_instance = _loggers[logger_name]
        logger_instance.setLevel(log_level)
    
    # Only log the change if the new level allows it
    if log_level <= logging.INFO:
        root_logger.info(f"🔄 Log level changed to: {level.upper()}")

# Convenience functions for common logging patterns
def log_reward_info(logger: logging.Logger, component: str, value: float, details: str = "") -> None:
    """Log reward information at INFO level"""
    logger.info(f"💰 {component}: {value:+.2f} {details}")

def log_route_info(logger: logging.Logger, message: str, distance: float = None) -> None:
    """Log route/navigation information at INFO level"""
    if distance is not None:
        logger.info(f"🛣️  {message} (distance: {distance:.1f}m)")
    else:
        logger.info(f"🛣️  {message}")

def log_prediction_debug(logger: logging.Logger, model_name: str, prediction: any) -> None:
    """Log model predictions at DEBUG level"""
    logger.debug(f"🤖 {model_name} prediction: {prediction}")

def log_penalty_warning(logger: logging.Logger, penalty_type: str, value: float, reason: str = "") -> None:
    """Log penalties at WARNING level"""
    logger.warning(f"⚠️  {penalty_type}: {value:+.2f} - {reason}")

def log_safety_warning(logger: logging.Logger, safety_issue: str, details: str = "") -> None:
    """Log safety violations at WARNING level"""
    logger.warning(f"🚨 SAFETY: {safety_issue} {details}")

def log_performance_info(logger: logging.Logger, metric: str, value: float, unit: str = "") -> None:
    """Log performance metrics at INFO level"""
    logger.info(f"📊 {metric}: {value:.2f} {unit}")

# Note: Logging is configured when first get_logger() is called or explicitly via configure_logging()