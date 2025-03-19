# src/imagefinder/raw_processing.py
import os
import time
import logging
from pathlib import Path
import numpy as np
import cv2

# Try to import the Rust module
try:
    import raw_processor
    RUST_ENABLED = True
    logging.info("Rust raw_processor module loaded successfully")
except ImportError:
    RUST_ENABLED = False
    logging.warning("Rust raw_processor module not available. Using Python implementation only.")

def convert_raw_to_jpg_and_load(path: str) -> np.ndarray:
    """
    Convert a RAW file to JPG and load it for hashing.
    Uses the Rust implementation if available, with Python fallbacks.
    """
    temp_dir = os.path.join(os.path.expanduser("~"), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_jpg = os.path.join(temp_dir, f"std_conv_{int(time.time()*1000000)}.jpg")
    
    try:
        # First try Rust direct grayscale conversion if available
        if RUST_ENABLED:
            try:
                start_time = time.time()
                img = raw_processor.rust_raw_to_grayscale(path)
                if img is not None and img.size > 0:
                    logging.debug(f"Rust direct grayscale conversion successful in {time.time() - start_time:.3f}s")
                    return img
            except Exception as e:
                logging.warning(f"Rust direct grayscale conversion failed: {e}, trying JPG method")
                
                # If direct conversion fails, try the JPG conversion method
                try:
                    start_time = time.time()
                    if raw_processor.rust_convert_raw_to_jpg(path, temp_jpg):
                        if os.path.isfile(temp_jpg) and os.path.getsize(temp_jpg) > 0:
                            img = cv2.imread(temp_jpg, cv2.IMREAD_GRAYSCALE)
                            if img is not None and img.size > 0:
                                logging.debug(f"Rust JPG conversion successful in {time.time() - start_time:.3f}s")
                                return img
                except Exception as e:
                    logging.warning(f"Rust JPG conversion failed: {e}, falling back to Python implementation")
        
        # Fall back to Python implementation - import them locally to avoid circular imports
        from imagefinder.scanner import (
            convert_cr3_with_exiftool,
            extract_preview_with_exiftool,
            convert_with_dcraw_auto_bright,
            convert_with_dcraw_camera_wb,
            convert_with_rawpy,
            is_raw_format
        )
        
        logging.debug("Using Python implementation for RAW conversion")
        
        # Special handling for CR3 files
        if Path(path).suffix.lower() == ".cr3":
            if convert_cr3_with_exiftool(path, temp_jpg):
                if os.path.isfile(temp_jpg) and os.path.getsize(temp_jpg) > 0:
                    img = cv2.imread(temp_jpg, cv2.IMREAD_GRAYSCALE)
                    if img is not None and img.size > 0:
                        return img
        
        # Try different conversion methods
        methods = [
            extract_preview_with_exiftool,
            convert_with_dcraw_auto_bright,
            convert_with_dcraw_camera_wb,
            convert_with_rawpy,
        ]
        
        last_error = None
        for method in methods:
            try:
                start_time = time.time()
                if method(path, temp_jpg):
                    if os.path.isfile(temp_jpg) and os.path.getsize(temp_jpg) > 0:
                        img = cv2.imread(temp_jpg, cv2.IMREAD_GRAYSCALE)
                        if img is not None and img.size > 0:
                            method_name = method.__name__
                            logging.debug(f"{method_name} successful in {time.time() - start_time:.3f}s")
                            return img
            except Exception as e:
                last_error = e
                continue
        
        # If all external tools fail, try direct rawpy processing as a final fallback
        try:
            import rawpy
            start_time = time.time()
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess()
                img = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                if img is not None and img.size > 0:
                    logging.debug(f"Direct rawpy processing successful in {time.time() - start_time:.3f}s")
                    return img
        except Exception as rawpy_error:
            last_error = rawpy_error
        
        raise ValueError(f"Failed to convert RAW to JPG: {last_error}")
    finally:
        if os.path.exists(temp_jpg):
            try:
                os.remove(temp_jpg)
            except:
                pass

def compute_average_hash(img: np.ndarray) -> str:
    """
    Compute average hash for image indexing.
    Uses the Rust implementation if available, with Python fallback.
    """
    # Resize to 8x8
    resized = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale if not already
    if len(resized.shape) > 2:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    
    # Use Rust implementation if available
    if RUST_ENABLED:
        try:
            start_time = time.time()
            hash_str = raw_processor.rust_compute_average_hash(gray)
            logging.debug(f"Rust average hash computation successful in {time.time() - start_time:.6f}s")
            return hash_str
        except Exception as e:
            logging.warning(f"Rust hash computation failed: {e}, falling back to Python")
    
    # Fall back to Python implementation
    start_time = time.time()
    avg_pixel_value = gray.mean()
    
    # Compute the hash
    hash_str = ""
    for i in range(8):
        for j in range(8):
            if gray[i, j] >= avg_pixel_value:
                hash_str += "1"
            else:
                hash_str += "0"
    
    logging.debug(f"Python average hash computation completed in {time.time() - start_time:.6f}s")
    return hash_str

def compute_perceptual_hash(img: np.ndarray) -> str:
    """
    Compute perceptual hash (pHash) for better matching.
    Uses the Rust implementation if available, with Python fallback.
    """
    # Resize to 32x32
    resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale if not already
    if len(resized.shape) > 2:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    
    # Use Rust implementation if available
    if RUST_ENABLED:
        try:
            start_time = time.time()
            hash_str = raw_processor.rust_compute_perceptual_hash(gray)
            logging.debug(f"Rust perceptual hash computation successful in {time.time() - start_time:.6f}s")
            return hash_str
        except Exception as e:
            logging.warning(f"Rust perceptual hash computation failed: {e}, falling back to Python")
    
    # Fall back to Python implementation
    start_time = time.time()
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
    
    logging.debug(f"Python perceptual hash computation completed in {time.time() - start_time:.6f}s")
    return hash_str
