import logging
import os
import sys
import psutil
import time
import functools
from datetime import datetime
from pathlib import Path
from rich.logging import RichHandler
from logging.handlers import RotatingFileHandler

def log_system_metrics(logger=None):
    """
    Logs current CPU and Memory usage.
    """
    if logger is None:
        logger = logging.getLogger("personalization_engine")
    
    cpu_usage = psutil.cpu_percent(interval=None)
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    
    logger.info(f"[Metrics] [MemoryFootprint] CPU Usage: {cpu_usage}% | Memory Usage: {memory_usage}% ({memory_info.used / (1024**2):.1f}MB used)")

def log_complexity(component: str, time_complexity: str, space_complexity: str):
    """
    Logs complexity observations for a system component.
    """
    logger = logging.getLogger("personalization_engine")
    logger.info(f"[Complexity] Component: {component} | Time: {time_complexity} | Space: {space_complexity}")

def log_execution_details(func):
    """
    Decorator to log function execution time, memory usage, and metadata.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("personalization_engine")
        
        # Start metrics
        start_time = time.time()
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
        
        # Log input metadata (simplified)
        func_name = func.__name__
        arg_info = f"args={len(args)} kwargs={list(kwargs.keys())}"
        logger.debug(f"[ExecutionStart] Function: {func_name} | {arg_info}")
        
        try:
            result = func(*args, **kwargs)
            
            # End metrics
            duration = time.time() - start_time
            mem_after = process.memory_info().rss / (1024 * 1024)
            mem_diff = mem_after - mem_before
            
            # Log output metadata
            res_meta = "N/A"
            try:
                import numpy as np
                if isinstance(result, np.ndarray):
                    res_meta = f"shape={result.shape}"
                elif hasattr(result, '__len__'):
                    res_meta = f"len={len(result)}"
            except ImportError:
                if hasattr(result, '__len__'):
                    res_meta = f"len={len(result)}"
            
            logger.info(f"[ExecutionComplete] {func_name} | Duration: {duration:.4f}s | MemDelta: {mem_diff:+.2f}MB | Result: {res_meta}")
            return result
        except Exception as e:
            logger.error(f"[ExecutionError] {func_name} failed: {e}", exc_info=True)
            raise
            
    return wrapper

def setup_logger(name="personalization_engine", log_file="app.log", level=logging.INFO):
    """
    Sets up a logger with both console (Rich) and file output.
    Ensures real-time output by disabling internal buffering where possible.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_path = log_dir / log_file
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
        
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    
    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_path=False
    )
    console_handler.setLevel(level)
    
    # Use standard FileHandler to ensure immediate writes to disk
    file_handler = RotatingFileHandler(
        log_path, 
        maxBytes=10*1024*1024, # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG) 
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.propagate = True
    
    return logger

def get_logger(module_name):
    if module_name.startswith("personalization_engine"):
        name = module_name
    else:
        parts = module_name.split('.')
        if "personalization_engine" in parts:
            name = module_name
        else:
            name = f"personalization_engine.{module_name}"
            
    return logging.getLogger(name)

main_logger = setup_logger("personalization_engine")
