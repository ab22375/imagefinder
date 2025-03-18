#!/usr/bin/env python3

import os
import time
import argparse
import sys
from pathlib import Path

# Import modules from the imagefinder package
from imagefinder.database import init_database as InitDatabase, open_database as OpenDatabase
from imagefinder.imageprocessor import find_similar_images as FindSimilarImages, SearchOptions
from imagefinder.mylogging import setup_logger as SetupLogger
from imagefinder.scanner import scan_and_store_folder as ScanAndStoreFolder, ScanOptions
from imagefinder.utils import get_default_database_path as GetDefaultDatabasePath, parse_threshold as ParseThreshold

def parse_arguments():
    """Parse command line arguments and return them as a dictionary"""
    parser = argparse.ArgumentParser(description="ImageFinder: Find similar images")
    
    # Add common arguments
    parser.add_argument("command", nargs='?', choices=["scan", "search"], help="Command to execute")
    parser.add_argument("--database", "--db", dest="database", help="Path to the database")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--logfile", help="Path to log file")
    parser.add_argument("--prefix", help="Source prefix for filtering")
    
    # Scan command arguments
    parser.add_argument("--folder", help="Folder path to scan for images")
    parser.add_argument("--force", action="store_true", help="Force rewrite existing entries")
    
    # Search command arguments
    parser.add_argument("--image", help="Query image path")
    parser.add_argument("--threshold", help="Similarity threshold (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Convert namespace to dictionary
    return vars(args)

def print_usage():
    """Print command usage"""
    print("Usage:")
    print("  imagefinder scan --folder=PATH [options]")
    print("  imagefinder search --image=PATH [options]")
    print("\nOptions:")
    print("  --database=PATH, --db=PATH   Specify database path")
    print("  --prefix=PREFIX              Source prefix for filtering")
    print("  --threshold=VALUE            Similarity threshold (0.0-1.0)")
    print("  --debug                      Enable debug mode")
    print("  --logfile=PATH               Specify log file path")
    print("  --force                      Force rewrite existing entries (scan only)")

def handle_scan_command(args, db_path, debug_mode):
    # Get folder path
    folder_path = args.get("folder")
    if not folder_path:
        print("Error: Missing folder path (use --folder=PATH)")
        sys.exit(1)
        
    # Get source prefix
    source_prefix = args.get("prefix", "")
    
    # Get force rewrite flag
    force_rewrite = args.get("force", False)
    
    # Verify folder path exists
    if not os.path.exists(folder_path):
        print(f"Folder path does not exist: {folder_path}")
        sys.exit(1)
        
    start_time = time.time()
    
    # Initialize database
    try:
        db = InitDatabase(db_path)
    except Exception as e:
        print(f"Error initializing database: {e}")
        sys.exit(1)
        
    try:
        # Scan folder and store image information
        scan_options = ScanOptions(
            folder_path=folder_path,
            source_prefix=source_prefix,
            force_rewrite=force_rewrite,
            debug_mode=debug_mode,
            db_path=db_path
        )
        
        ScanAndStoreFolder(db, scan_options)
    except Exception as e:
        print(f"Error scanning folder: {e}")
        sys.exit(1)
    finally:
        db.close()
        
    # Print execution time
    duration = time.time() - start_time
    print(f"\nTotal execution time: {duration:.2f} seconds")

def handle_search_command(args, db_path, debug_mode):
    # Get query image path
    query_path = args.get("image")
    if not query_path:
        print("Error: Missing query image path (use --image=PATH)")
        sys.exit(1)
        
    # Set custom threshold if provided
    threshold = 0.8  # Default threshold
    if threshold_str := args.get("threshold"):
        try:
            threshold, _ = ParseThreshold(threshold_str)  # Using the tuple return value from parse_threshold
        except Exception as e:
            print(f"Warning: {e}")
            
    # Get source prefix for filtering
    source_prefix = args.get("prefix", "")
    
    # Verify paths exist
    if not os.path.exists(query_path):
        print(f"Query image does not exist: {query_path}")
        sys.exit(1)
        
    if not os.path.exists(db_path):
        print(f"Database does not exist: {db_path}. Run scan command first.")
        sys.exit(1)
        
    start_time = time.time()
    
    # Open database
    try:
        db = OpenDatabase(db_path)
    except Exception as e:
        print(f"Error opening database: {e}")
        sys.exit(1)
        
    try:
        print("Searching for similar images...")
        if source_prefix:
            print(f"Filtering by source prefix: {source_prefix}")
            
        # Find similar images
        search_options = SearchOptions(
            query_path=query_path,
            threshold=threshold,
            source_prefix=source_prefix,
            debug_mode=debug_mode
        )
        
        matches = FindSimilarImages(db, search_options)
        
        # Print top matches
        print("\nTop Matches:")
        limit = 5  # Show top 5 matches
        
        if not matches:
            print("No matches found.")
        else:
            for i, match in enumerate(matches[:limit]):
                print(f"{i+1}. Image: {match.path}")
                if match.source_prefix:
                    print(f"   Source: {match.source_prefix}")
                print(f"   SSIM Score: {match.ssim_score:.4f}")
    except Exception as e:
        print(f"Error finding similar images: {e}")
        sys.exit(1)
    finally:
        db.close()
        
    # Print execution time
    duration = time.time() - start_time
    print(f"\nTotal search time: {duration:.2f} seconds")

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Get the command
    command = args.get("command")
    
    # Set default database path
    db_path = GetDefaultDatabasePath()
    if custom_db := args.get("database"):
        db_path = custom_db
    
    # Setup debug logging if enabled
    debug_mode = args.get("debug", False)
    if debug_mode:
        log_path = "imagefinder.log"
        if custom_log_path := args.get("logfile"):
            log_path = custom_log_path
        try:
            SetupLogger(log_path)
            print(f"Debug mode enabled. Logging to: {log_path}")
        except Exception as e:
            print(f"Warning: Failed to setup logging: {e}")
    
    # Check if required arguments are missing
    show_usage = False
    
    if not command:
        show_usage = True
    elif command == "scan" and not args.get("folder"):
        show_usage = True
    elif command == "search" and not args.get("image"):
        show_usage = True
    
    # Show usage if required arguments are missing
    if show_usage:
        print_usage()
        sys.exit(1)
    
    # Handle commands
    if command == "scan":
        handle_scan_command(args, db_path, debug_mode)
    elif command == "search":
        handle_search_command(args, db_path, debug_mode)
    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)

if __name__ == "__main__":
    main()