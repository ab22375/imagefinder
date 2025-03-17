#!/usr/bin/env python3

import os
import time
import subprocess
import sqlite3
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import threading
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

# Import local modules with relative imports
from imagefinder.database import check_image_exists, store_image_info
from imagefinder.imageprocessor import load_image, find_similar_images, ImageLoaderRegistry
from imagefinder.image_types import ImageInfo, ImageMatch

@dataclass
class ScanOptions:
    """Options for image scanning"""
    folder_path: str
    source_prefix: str
    force_rewrite: bool
    debug_mode: bool
    db_path: str

@dataclass
class ProcessImageResult:
    """Result of processing an image"""
    path: str
    success: bool
    error: Optional[str] = None

def process_and_store_image(db_conn: sqlite3.Connection, path: str, source_prefix: str, options: ScanOptions) -> ProcessImageResult:
    """Process a single image and store it in the database"""
    result = ProcessImageResult(
        path=path,
        success=False
    )

    # Skip processing if the image already exists and hasn't been modified
    if not options.force_rewrite:
        try:
            exists, stored_mod_time = check_image_exists(db_conn, path, source_prefix)
            
            if exists:
                # Image already indexed, check if it needs update
                try:
                    file_info = os.stat(path)
                except OSError as e:
                    result.error = f"Cannot stat file {path}: {str(e)}"
                    return result

                # Parse stored time and compare with file modified time
                try:
                    stored_time = time.strptime(stored_mod_time, "%Y-%m-%dT%H:%M:%S%z")
                    file_mod_time = time.localtime(file_info.st_mtime)
                    
                    # If file hasn't been modified, skip processing
                    if not file_mod_time > stored_time:
                        if options.debug_mode:
                            logging.debug(f"Skipping unchanged image: {path}")
                        result.success = True
                        return result
                except ValueError as e:
                    result.error = f"Cannot parse stored time for {path}: {str(e)}"
                    return result
        except Exception as e:
            result.error = f"Database error for {path}: {str(e)}"
            return result

    # Get file info
    try:
        file_info = os.stat(path)
    except OSError as e:
        result.error = f"Cannot stat file {path}: {str(e)}"
        return result

    # Get file format from extension (without the dot)
    file_format = Path(path).suffix.lower().lstrip('.')

    # Detect if this is a RAW image
    is_raw_image = is_raw_format(path)

    # Load and process the image - for RAW files, convert to JPG first
    try:
        if is_raw_image:
            if options.debug_mode:
                logging.debug(f"Converting RAW image to JPG for consistent hashing: {path}")

            # First try our dedicated RAW to JPG conversion
            try:
                img = convert_raw_to_jpg_and_load(path)
                if options.debug_mode:
                    logging.debug(f"Successfully converted RAW to JPG for: {path}")
            except Exception as e:
                if options.debug_mode:
                    logging.warning(f"RAW to JPG conversion failed: {e}, falling back to standard loader")
                img = load_image(path)
        else:
            # For non-RAW files, load normally
            img = load_image(path)

        # Make sure the image loaded successfully
        if img is None or img.size == 0:
            result.error = f"Failed to load image {path}: Image is empty"
            return result

        # Compute hashes
        avg_hash = compute_average_hash(img)
        p_hash = compute_perceptual_hash(img)

        # Log hash information for debugging raw images
        if options.debug_mode and is_raw_image:
            logging.debug(f"RAW image hashes - {path} - avgHash: {avg_hash}, pHash: {p_hash}")

        # Create ImageInfo object
        image_info = ImageInfo(
            id=0,  # Will be assigned by the database
            path=path,
            source_prefix=source_prefix,
            format=file_format,
            width=img.shape[1],  # Cols
            height=img.shape[0],  # Rows
            created_at="",  # Will be set by database
            modified_at=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(file_info.st_mtime)),
            size=file_info.st_size,
            average_hash=avg_hash,
            perceptual_hash=p_hash,
            is_raw_format=is_raw_image
        )

        # Store in database
        store_image_info(db_conn, image_info, options.force_rewrite)

        if options.debug_mode and is_raw_image:
            logging.debug(f"Successfully indexed RAW image: {path}")

        result.success = True
        return result

    except Exception as e:
        result.error = f"Error processing {path}: {str(e)}"
        return result

def compute_average_hash(img: np.ndarray) -> str:
    """Compute average hash for image indexing"""
    # Resize to 8x8
    resized = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale if not already
    if len(resized.shape) > 2:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    
    # Calculate average pixel value
    avg_pixel_value = gray.mean()
    
    # Compute the hash
    hash_str = ""
    for i in range(8):
        for j in range(8):
            if gray[i, j] >= avg_pixel_value:
                hash_str += "1"
            else:
                hash_str += "0"
    
    return hash_str

def compute_perceptual_hash(img: np.ndarray) -> str:
    """Compute perceptual hash (pHash) for better matching"""
    # Resize to 32x32
    resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale if not already
    if len(resized.shape) > 2:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    
    # Create a simplified hash based on brightness patterns
    regions = 8  # 8x8 regions
    region_values = []
    
    # Calculate average brightness in each region
    region_height = gray.shape[0] // regions
    region_width = gray.shape[1] // regions
    
    for i in range(regions):
        for j in range(regions):
            # Calculate region boundaries
            start_y = i * region_height
            end_y = (i + 1) * region_height
            start_x = j * region_width
            end_x = (j + 1) * region_width
            
            # Calculate average for region
            region = gray[start_y:end_y, start_x:end_x]
            avg = np.mean(region)
            region_values.append(avg)
    
    # Calculate median
    median = np.median(region_values)
    
    # Create hash based on whether each value is above median
    hash_str = ""
    for val in region_values:
        if val > median:
            hash_str += "1"
        else:
            hash_str += "0"
    
    return hash_str

def convert_raw_to_jpg_and_load(path: str) -> np.ndarray:
    """Convert a RAW file to JPG and load it for hashing"""
    temp_dir = os.path.join(os.path.expanduser("~"), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_jpg = os.path.join(temp_dir, f"std_conv_{int(time.time()*1000000)}.jpg")
    
    try:
        # Special handling for CR3 files
        if Path(path).suffix.lower() == ".cr3":
            if convert_cr3_with_exiftool(path, temp_jpg):
                # Check if the file was created successfully
                if os.path.isfile(temp_jpg) and os.path.getsize(temp_jpg) > 0:
                    # Load the standard JPG representation
                    img = cv2.imread(temp_jpg, cv2.IMREAD_GRAYSCALE)
                    if img is not None and img.size > 0:
                        return img

        # Try different conversion methods in order of preference
        methods = [
            extract_preview_with_exiftool,    # Extract embedded preview (best match for camera JPGs)
            convert_with_dcraw_auto_bright,   # Use dcraw with auto-brightness
            convert_with_dcraw_camera_wb,     # Use dcraw with camera white balance
        ]

        last_error = None
        for method in methods:
            try:
                if method(path, temp_jpg):
                    # Check if the file was created successfully
                    if os.path.isfile(temp_jpg) and os.path.getsize(temp_jpg) > 0:
                        # Load the standard JPG representation
                        img = cv2.imread(temp_jpg, cv2.IMREAD_GRAYSCALE)
                        if img is not None and img.size > 0:
                            return img
            except Exception as e:
                last_error = e
                continue

        # If all methods fail, raise the error
        raise ValueError(f"Failed to convert RAW to JPG: {last_error}")
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_jpg):
            try:
                os.remove(temp_jpg)
            except:
                pass

def extract_preview_with_exiftool(path: str, output_path: str) -> bool:
    """Extract the embedded preview JPEG from the RAW file using exiftool"""
    try:
        # Use exiftool to extract the preview image
        # -b = output in binary mode
        # -PreviewImage = extract the preview image
        process = subprocess.run(
            ["exiftool", "-b", "-PreviewImage", "-w", output_path, path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        return process.returncode == 0
    except Exception:
        return False

def convert_with_dcraw_auto_bright(path: str, output_path: str) -> bool:
    """Convert using dcraw with auto-brightness, which often matches camera output"""
    try:
        # -w = use camera white balance
        # -a = auto-brightness (mimics camera)
        # -q 3 = high-quality interpolation
        # -O = output to specified file
        process = subprocess.run(
            ["dcraw", "-w", "-a", "-q", "3", "-O", output_path, path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        return process.returncode == 0
    except Exception:
        return False

def convert_with_dcraw_camera_wb(path: str, output_path: str) -> bool:
    """Convert using dcraw with camera white balance, no auto-brightness"""
    try:
        # -w = use camera white balance
        # -q 3 = high-quality interpolation
        # -O = output to specified file
        process = subprocess.run(
            ["dcraw", "-w", "-q", "3", "-O", output_path, path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        return process.returncode == 0
    except Exception:
        return False

def is_raw_format(path: str) -> bool:
    """Check if a file is in RAW format"""
    ext = Path(path).suffix.lower()
    raw_formats = [".dng", ".raf", ".arw", ".nef", ".cr2", ".cr3", ".nrw", ".srf"]
    return ext in raw_formats

def convert_cr3_with_exiftool(path: str, output_path: str) -> bool:
    """Specialized function for CR3 files which often need special handling"""
    try:
        # CR3 files often have multiple preview images
        # Try extracting the largest preview image
        process = subprocess.run(
            ["exiftool", "-b", "-LargePreviewImage", "-w", output_path, path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )

        # Check if the output file was created and has content
        if process.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True

        # Try alternative tags for preview images
        tags = [
            "PreviewImage",
            "OtherImage",
            "ThumbnailImage",
            "FullPreviewImage",
        ]

        for tag in tags:
            process = subprocess.run(
                ["exiftool", "-b", f"-{tag}", "-w", output_path, path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )

            # Check if successful
            if process.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return True

        return False
    except Exception:
        return False

def scan_and_store_folder(db_conn: sqlite3.Connection, options: ScanOptions) -> None:
    """Scan a folder and store image information in the database"""
    # Prepare registry for file type checking
    registry = ImageLoaderRegistry()
    
    # Count total files before starting
    total_files = 0
    raw_files = 0
    
    if options.debug_mode:
        logging.debug(f"Starting image scan on folder: {options.folder_path}")
        logging.debug(f"Force rewrite: {options.force_rewrite}, Source prefix: {options.source_prefix}")
    
    # First pass to count files
    for root, _, files in os.walk(options.folder_path):
        for file in files:
            path = os.path.join(root, file)
            
            # Check if any loader can handle this file
            if registry.can_load_file(path):
                total_files += 1
                # Count RAW images separately
                if is_raw_format(path):
                    raw_files += 1
    
    print(f"Starting image indexing...\nTotal image files to process: {total_files} (including {raw_files} RAW files)")
    print(f"Force rewrite mode: {options.force_rewrite}")
    if options.source_prefix:
        print(f"Source prefix: {options.source_prefix}")
    if options.debug_mode:
        print("Debug mode: enabled")
        logging.debug(f"Found {total_files} image files to process ({raw_files} RAW files)")
    
    # Variables for tracking progress
    processed = 0
    errors = 0
    raw_processed = 0
    raw_errors = 0
    mutex = Lock()
    
    # Progress display thread
    def progress_display():
        while processed < total_files:
            with mutex:
                if errors > 0:
                    print(f"\rProgress: {processed}/{total_files} (Errors: {errors}, RAW: {raw_processed}/{raw_files})", end="")
                else:
                    print(f"\rProgress: {processed}/{total_files} (RAW: {raw_processed}/{raw_files})", end="")
            time.sleep(0.5)
    
    # Start progress thread
    progress_thread = threading.Thread(target=progress_display)
    progress_thread.daemon = True
    progress_thread.start()
    
    # Process files with ThreadPoolExecutor
    start_time = time.time()

    def process_file(path):
        nonlocal processed, errors, raw_processed, raw_errors
        
        # Create a new connection for each thread
        thread_db_conn = sqlite3.connect(options.db_path)
        try:
            result = process_and_store_image(thread_db_conn, path, options.source_prefix, options)
        
            with mutex:
                processed += 1
                
                # Check if this is a RAW file
                if is_raw_format(path):
                    raw_processed += 1
                    if not result.success:
                        raw_errors += 1
                
                if not result.success:
                    errors += 1
                    if options.debug_mode:
                        logging.error(f"Error processing image {path}: {result.error}")
                elif options.debug_mode:
                    logging.debug(f"Successfully processed image: {path}")
        finally:
            thread_db_conn.close()
                
    # Collect paths to process
    paths_to_process = []
    for root, _, files in os.walk(options.folder_path):
        for file in files:
            path = os.path.join(root, file)
            if registry.can_load_file(path):
                paths_to_process.append(path)
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(process_file, paths_to_process)
    
    # Final output
    elapsed = time.time() - start_time
    print("\nIndexing complete.")
    
    # Log final statistics
    if options.debug_mode:
        logging.debug(f"Scan completed in {elapsed:.2f}s. Processed: {processed}, Errors: {errors}, "
                     f"RAW files: {raw_processed}, RAW errors: {raw_errors}")
    
    print(f"Processed {processed} images in {int(elapsed)} seconds.")
    if raw_processed > 0:
        print(f"Successfully processed {raw_processed-raw_errors}/{raw_files} RAW image files.")
    
    if errors > 0:
        print(f"Encountered {errors} errors during indexing.")
        print("Check the log file for details.")

# Define aliases for compatibility with original Go code
ScanAndStoreFolder = scan_and_store_folder