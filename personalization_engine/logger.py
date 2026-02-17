
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from rich.logging import RichHandler
from logging.handlers import RotatingFileHandler

def setup_logger(name="personalization_engine", log_file="app.log", level=logging.INFO):
    """
    Sets up a logger with both console (Rich) and file output.
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_path = log_dir / log_file
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if logger already has handlers to avoid duplicate logs
    if logger.handlers:
        return logger
        
    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    
    # 1. Console Handler using Rich for beautiful output
    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_path=False
    )
    console_handler.setLevel(level)
    
    # 2. File Handler for persistent logs
    file_handler = RotatingFileHandler(
        log_path, 
        maxBytes=10*1024*1024, # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG) # Always log DEBUG to file
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Allow propagation so sub-loggers work correctly
    logger.propagate = True
    
    return logger

# Convenience function to get a logger for a specific module
def get_logger(module_name):
    """
    Returns a child logger for the given module name.
    If the name doesn't start with 'personalization_engine.', it will be prefixed.
    """
    if module_name.startswith("personalization_engine"):
        name = module_name
    else:
        # Handle cases like __name__ which might be just 'qwen_adapter' or 'personalization_engine.qwen_adapter'
        parts = module_name.split('.')
        if "personalization_engine" in parts:
            name = module_name
        else:
            name = f"personalization_engine.{module_name}"
            
    return logging.getLogger(name)

# Initialize the root logger for the project
# This ensures that any logger starting with 'personalization_engine' is configured
main_logger = setup_logger("personalization_engine")
