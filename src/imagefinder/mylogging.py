#!/usr/bin/env python3

import os
import logging
import threading
import time
from datetime import datetime
from typing import Optional, Any

# Module-level variables
logger = None
log_file_handler = None
log_mutex = threading.Lock()
is_setup = False

def setup_logger(log_file_path: str) -> Optional[Exception]:
    """
    Initialize the debug logger with the specified log file
    
    Args:
        log_file_path: Path where the log file will be created
        
    Returns:
        Exception if there was an error, None otherwise
    """
    global logger, log_file_handler, is_setup
    
    with log_mutex:
        # Check if logger is already set up
        if is_setup:
            return None
        
        try:
            # Configure the logger
            logger = logging.getLogger('ImageFinder')
            logger.setLevel(logging.DEBUG)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            
            # Create file handler
            log_file_handler = logging.FileHandler(log_file_path)
            log_file_handler.setLevel(logging.DEBUG)
            log_file_handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(log_file_handler)
            
            # Log startup information
            logger.info(f"--- ImageFinder Debug Log Started at {datetime.now().isoformat()} ---")
            
            is_setup = True
            return None
            
        except Exception as e:
            return Exception(f"Failed to open log file: {str(e)}")

def close_logger() -> None:
    """
    Close the log file
    """
    global logger, log_file_handler, is_setup
    
    with log_mutex:
        if is_setup and log_file_handler is not None:
            logger.info(f"--- ImageFinder Debug Log Closed at {datetime.now().isoformat()} ---")
            log_file_handler.close()
            logger.removeHandler(log_file_handler)
            log_file_handler = None
            is_setup = False

def debug_log(format_str: str, *args: Any) -> None:
    """
    Log a debug message if debug mode is enabled
    
    Args:
        format_str: Format string
        *args: Arguments for format string
    """
    with log_mutex:
        if is_setup and logger is not None:
            if args:
                logger.debug(format_str % args)
            else:
                logger.debug(format_str)

def log_error(format_str: str, *args: Any) -> None:
    """
    Log an error message
    
    Args:
        format_str: Format string
        *args: Arguments for format string
    """
    with log_mutex:
        if is_setup and logger is not None:
            if args:
                logger.error(format_str % args)
            else:
                logger.error(format_str)

def log_warning(format_str: str, *args: Any) -> None:
    """
    Log a warning message
    
    Args:
        format_str: Format string
        *args: Arguments for format string
    """
    with log_mutex:
        if is_setup and logger is not None:
            if args:
                logger.warning(format_str % args)
            else:
                logger.warning(format_str)

def log_image_processed(path: str, success: bool, err_msg: str = None) -> None:
    """
    Log when an image is processed
    
    Args:
        path: Path to the image
        success: Whether the processing was successful
        err_msg: Error message if processing failed
    """
    with log_mutex:
        if is_setup and logger is not None:
            if success:
                logger.info(f"PROCESSED: {path}")
            else:
                logger.error(f"FAILED: {path} - Error: {err_msg}")