#!/usr/bin/env python3

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple

def parse_arguments() -> Dict[str, str]:
    """
    Converts command-line arguments into a dictionary of flags and values.
    
    Returns:
        Dict[str, str]: Dictionary containing parsed command-line arguments
    """
    args = {}
    
    # First, identify the command (scan/search)
    command = ""
    command_index = -1
    
    for i in range(1, len(sys.argv)):
        if sys.argv[i] in ["scan", "search"]:
            command = sys.argv[i]
            command_index = i
            break
    
    if command:
        args["command"] = command
    
    # Process all arguments, skipping the command
    i = 1
    while i < len(sys.argv):
        if i == command_index:
            i += 1
            continue
        
        arg = sys.argv[i]
        
        # Handle flags with equals sign (--key=value)
        if arg.startswith("--") and "=" in arg:
            parts = arg.split("=", 1)
            flag_name = parts[0][2:]  # Remove '--' prefix
            args[flag_name] = parts[1]
            i += 1
            continue
        
        # Handle flags without equals sign (--key value)
        if arg.startswith("--"):
            flag_name = arg[2:]  # Remove '--' prefix
            
            # Check if this is a boolean flag (no value)
            if i + 1 >= len(sys.argv) or sys.argv[i + 1].startswith("--"):
                args[flag_name] = "true"
                i += 1
            else:
                # The next argument is the value
                args[flag_name] = sys.argv[i + 1]
                i += 2  # Skip the value in the next iteration
        else:
            i += 1
    
    return args

def get_default_database_path() -> str:
    """
    Returns the default path for the database file.
    
    Returns:
        str: Path to the default database file
    """
    try:
        # Get the executable path
        exe_path = sys.executable
        # Get the directory containing the executable
        exe_dir = os.path.dirname(exe_path)
        # Return the default database path in the same directory
        return os.path.join(exe_dir, "images.db")
    except:
        # Fallback to current directory if executable path can't be determined
        return "images.db"

def print_usage() -> None:
    """
    Outputs the command-line usage instructions.
    """
    program_name = os.path.basename(sys.argv[0])
    
    print(f"Usage:")
    print(f"  {program_name} scan --folder=PATH [--database=PATH] [--prefix=NAME] [--force] [--debug] [--logfile=PATH]")
    print(f"  {program_name} search --image=PATH [--database=PATH] [--threshold=VALUE] [--prefix=NAME] [--debug] [--logfile=PATH]")
    print(f"\nParameters:")
    print(f"  --folder      : Path to folder containing images to scan")
    print(f"  --image       : Path to query image for search")
    print(f"  --database    : Path to database file (default: {get_default_database_path()})")
    print(f"  --prefix      : Source prefix for scanning/filtering results")
    print(f"  --force       : Force rewrite existing entries during scan")
    print(f"  --threshold   : Similarity threshold for search (0.0-1.0, default: 0.8)")
    print(f"  --debug       : Enable debug mode (logs detailed information)")
    print(f"  --logfile     : Specify custom log file path (default: imagefinder.log)")
    print(f"\nExamples:")
    print(f"  {program_name} scan --folder=/path/to/images --prefix=ExternalDrive1 --debug")
    print(f"  {program_name} search --image=/path/to/query.jpg --threshold=0.85")

def parse_threshold(threshold_str: str) -> Tuple[float, str]:
    """
    Parses and validates the threshold value from string.
    
    Args:
        threshold_str (str): String representation of the threshold value
    
    Returns:
        Tuple[float, str]: The parsed threshold value and an error message if invalid
    """
    try:
        parsed_threshold = float(threshold_str)
        if parsed_threshold < 0 or parsed_threshold > 1:
            return 0.8, f"Invalid threshold value '{threshold_str}', using default (0.8)"
        return parsed_threshold, None
    except ValueError:
        return 0.8, f"Invalid threshold value '{threshold_str}', using default (0.8)"

def setup_logging(log_file_path: str = None, debug_mode: bool = False) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_file_path (str, optional): Path to log file. Defaults to None.
        debug_mode (bool, optional): Enable debug logging. Defaults to False.
    """
    log_level = logging.DEBUG if debug_mode else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Log to console
        ]
    )
    
    # Add file handler if log file path is provided
    if log_file_path:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    # Log startup information
    logging.info("Starting Image Finder")
    if debug_mode:
        logging.debug("Debug mode enabled")