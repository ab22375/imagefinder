#!/usr/bin/env python3

import os
import sys
import time
import subprocess
import tempfile
from image_types import ImageInfo, ImageMatch
import abc
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import cv2
import sqlite3
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchOptions:
    """Options for image searching"""
    query_path: str
    threshold: float
    source_prefix: str
    debug_mode: bool


@dataclass
class ImageMatch:
    """Represents a matching image with its score"""
    path: str
    source_prefix: str
    ssim_score: float


class ImageLoader(abc.ABC):
    """Interface for loading different image formats"""
    
    @abc.abstractmethod
    def can_load(self, path: str) -> bool:
        """Check if this loader can load the given file"""
        pass
    
    @abc.abstractmethod
    def load_image(self, path: str) -> np.ndarray:
        """Load an image and return it as a numpy array"""
        pass


class DefaultImageLoader(ImageLoader):
    """Handles common formats supported by OpenCV directly"""
    
    def can_load(self, path: str) -> bool:
        ext = Path(path).suffix.lower()
        # Check extension and make sure file exists and is readable
        if ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
            return os.path.isfile(path)
        return False
    
    def load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            raise ValueError(f"Failed to load image with default loader: {path}")
        return img


class RawImageLoader(ImageLoader):
    """Handles RAW camera formats"""
    
    def __init__(self):
        # Create a temp directory for raw image processing if needed
        self.temp_dir = tempfile.gettempdir()
    
    def can_load(self, path: str) -> bool:
        ext = Path(path).suffix.lower()
        # Explicitly include all requested formats: DNG, RAF, ARW, NEF, CR2, CR3
        raw_formats = ['.dng', '.raf', '.arw', '.nef', '.cr2', '.cr3', '.nrw', '.srf']
        if ext in raw_formats:
            return os.path.isfile(path)
        return False
    
    def load_image(self, path: str) -> np.ndarray:
        # Create a unique temporary filename for the converted image
        temp_filename = os.path.join(self.temp_dir, f"raw_conv_{int(time.time()*1000000)}.tiff")
        
        try:
            # Check if it's a CR3 file specifically
            if Path(path).suffix.lower() == '.cr3':
                success, img = self.try_cr3(path, temp_filename)
                if success:
                    return img
            
            # First try with dcraw
            success, img = self.try_dcraw(path, temp_filename)
            if success:
                return img
            
            # If dcraw fails, try libraw fallback
            success, img = self.try_libraw(path, temp_filename)
            if success:
                return img
            
            # If all else fails, attempt direct load (unlikely to work for most RAW formats)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None and img.size > 0:
                return img
            
            raise ValueError(f"Failed to load RAW image: {path} (all conversion methods failed)")
        
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except:
                    pass
    
    def try_dcraw(self, path: str, temp_filename: str) -> Tuple[bool, np.ndarray]:
        # Convert RAW to TIFF using dcraw
        # -T = output TIFF
        # -c = output to stdout (we redirect to file)
        # -w = use camera white balance
        # -q 3 = use high-quality interpolation
        try:
            with open(temp_filename, 'wb') as out_file:
                process = subprocess.Popen(
                    ['dcraw', '-T', '-c', '-w', '-q', '3', path],
                    stdout=out_file,
                    stderr=subprocess.PIPE
                )
                _, stderr = process.communicate()
                
                if process.returncode != 0:
                    logger.warning(f"dcraw conversion failed: {stderr.decode()}")
                    return False, None
            
            # Load the converted TIFF
            img = cv2.imread(temp_filename, cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                return False, None
            
            return True, img
        
        except Exception as e:
            logger.warning(f"Error during dcraw conversion: {e}")
            return False, None
    
    def try_libraw(self, path: str, temp_filename: str) -> Tuple[bool, np.ndarray]:
        # Try with rawtherapee-cli as an alternative for RAW conversion
        try:
            process = subprocess.Popen(
                ['rawtherapee-cli', '-o', temp_filename, '-c', path],
                stderr=subprocess.PIPE
            )
            _, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.warning(f"rawtherapee conversion failed: {stderr.decode()}")
                return False, None
            
            img = cv2.imread(temp_filename, cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                return False, None
            
            return True, img
        
        except Exception as e:
            logger.warning(f"Error during rawtherapee conversion: {e}")
            return False, None
    
    def try_cr3(self, path: str, temp_filename: str) -> Tuple[bool, np.ndarray]:
        # Try with exiftool to extract preview image (often works for CR3)
        try:
            with open(temp_filename, 'wb') as out_file:
                process = subprocess.Popen(
                    ['exiftool', '-b', '-PreviewImage', path],
                    stdout=out_file,
                    stderr=subprocess.PIPE
                )
                _, stderr = process.communicate()
            
            # Check if file has content
            if os.path.getsize(temp_filename) > 0:
                img = cv2.imread(temp_filename, cv2.IMREAD_GRAYSCALE)
                if img is not None and img.size > 0:
                    return True, img
            
            # Alternative approach using newer versions of libraw
            process = subprocess.Popen(
                ['libraw_unpack', '-O', temp_filename, path],
                stderr=subprocess.PIPE
            )
            _, stderr = process.communicate()
            
            if process.returncode == 0:
                img = cv2.imread(temp_filename, cv2.IMREAD_GRAYSCALE)
                if img is not None and img.size > 0:
                    return True, img
            
            return False, None
        
        except Exception as e:
            logger.warning(f"Error during CR3 conversion: {e}")
            return False, None


class HeicImageLoader(ImageLoader):
    """Handles HEIC/HEIF formats"""
    
    def can_load(self, path: str) -> bool:
        ext = Path(path).suffix.lower()
        if ext in ['.heic', '.heif']:
            return os.path.isfile(path)
        return False
    
    def load_image(self, path: str) -> np.ndarray:
        # HEIC typically needs conversion
        # First attempt to use an external converter like heif-convert
        temp_filename = os.path.join(tempfile.gettempdir(), f"heic_conv_{int(time.time()*1000000)}.jpg")
        
        try:
            # Try using heif-convert if available
            try:
                subprocess.run(['heif-convert', path, temp_filename], check=True)
                img = cv2.imread(temp_filename, cv2.IMREAD_GRAYSCALE)
                if img is not None and img.size > 0:
                    return img
            except (subprocess.SubprocessError, FileNotFoundError):
                # If heif-convert fails or isn't available, try direct reading
                pass
                
            # Try direct reading (may work with newer OpenCV versions)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None and img.size > 0:
                return img
                
            raise ValueError(f"Failed to load HEIC image: {path}")
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except:
                    pass


class ImageLoaderRegistry:
    """Manages available image loaders"""
    
    def __init__(self):
        self.loaders = [
            DefaultImageLoader(),
            RawImageLoader(),
            HeicImageLoader()
        ]
    
    def register_loader(self, loader: ImageLoader) -> None:
        """Add a custom loader to the registry"""
        self.loaders.append(loader)
    
    def get_loaders(self) -> List[ImageLoader]:
        """Returns the list of registered loaders"""
        return self.loaders
    
    # Changed to lowercase method name to match the method definitions
    def can_load_file(self, path: str) -> bool:
        """Check if any registered loader can handle the given file"""
        return any(loader.can_load(path) for loader in self.loaders)

    def load_image(self, path: str) -> np.ndarray:
        """Try to load an image using the appropriate loader"""
        for loader in self.loaders:
            if loader.can_load(path):
                return loader.load_image(path)  # Changed from LoadImage to load_image
        
        raise ValueError(f"No suitable loader found for image: {path}")

def load_image(path: str) -> np.ndarray:
    """Load an image in grayscale with error handling"""
    registry = ImageLoaderRegistry()
    return registry.load_image(path)


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


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute a simplified and more robust SSIM implementation"""
    # Check for valid matrices
    if img1 is None or img2 is None or img1.size == 0 or img2.size == 0:
        return 0.0
    
    # Convert to 8-bit grayscale if needed
    if img1.dtype != np.uint8:
        img1_gray = img1.astype(np.uint8)
    else:
        img1_gray = img1
    
    if img2.dtype != np.uint8:
        img2_gray = img2.astype(np.uint8)
    else:
        img2_gray = img2
    
    # Ensure images are same size
    resized = cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Calculate simple mean difference
    diff = cv2.absdiff(img1_gray, resized)
    
    if diff.size == 0:
        return 0.0
    
    mean_diff = np.mean(diff)
    
    if mean_diff > 255.0:
        return 0.0
    
    # Return similarity score (1 = identical, 0 = completely different)
    return 1.0 - (mean_diff / 255.0)


def calculate_hamming_distance(hash1: str, hash2: str) -> int:
    """Calculate the number of differing bits between two hash strings"""
    distance = 0
    min_len = min(len(hash1), len(hash2))
    
    for i in range(min_len):
        if hash1[i] != hash2[i]:
            distance += 1
    
    return distance


def is_jpg_format(path: str) -> bool:
    """Check if a file is in JPG format"""
    ext = Path(path).suffix.lower()
    return ext in ['.jpg', '.jpeg']


def get_base_filename(path: str) -> str:
    """Extract the base filename without extension and path"""
    # Get just the filename without the directory
    filename = os.path.basename(path)
    # Remove the extension
    return os.path.splitext(filename)[0]


def extract_digits(s: str) -> str:
    """Extract just the digits from a string"""
    return ''.join(ch for ch in s if ch.isdigit())


def are_filenames_related(path1: str, path2: str) -> bool:
    """Check if two filenames are likely related (e.g., IMG_1234.NEF and IMG_1234.JPG)"""
    base1 = get_base_filename(path1)
    base2 = get_base_filename(path2)
    
    # Direct match
    if base1 == base2:
        return True
    
    # Check for common patterns where JPG exports get renamed
    # 1. Some software adds suffixes like "_edited" or "-edited"
    if base1.startswith(base2) or base2.startswith(base1):
        return True
    
    # 2. Check for the same numeric part (cameras often use numeric names)
    digits1 = extract_digits(base1)
    digits2 = extract_digits(base2)
    
    if digits1 and digits2 and digits1 == digits2:
        return True
    
    return False


def is_raw_format(path: str) -> bool:
    """Check if a file is in RAW format"""
    ext = Path(path).suffix.lower()
    raw_formats = ['.dng', '.raf', '.arw', '.nef', '.cr2', '.cr3', '.nrw', '.srf']
    return ext in raw_formats


def find_similar_images(db_conn: sqlite3.Connection, options: SearchOptions) -> List[ImageMatch]:
    """Find similar images to the query image with enhanced RAW/JPG matching"""
    if options.debug_mode:
        logger.debug(f"Starting image search for: {options.query_path}")
        logger.debug(f"Threshold: {options.threshold:.2f}, Source Prefix: {options.source_prefix}")
    
    # Check if query is a RAW file
    is_raw_query = is_raw_format(options.query_path)
    if is_raw_query and options.debug_mode:
        logger.debug("Query image is a RAW format file, using special processing")
    
    # If the query is a JPG, check if there might be RAW versions to match against
    is_jpg_query = is_jpg_format(options.query_path)
    
    # Load the query image
    try:
        registry = ImageLoaderRegistry()
        query_img = None
        
        for loader in registry.get_loaders():
            if loader.can_load(options.query_path):
                query_img = loader.load_image(options.query_path)
                break
                
        if query_img is None:
            return []
            
        # Compute hashes for query image
        avg_hash = compute_average_hash(query_img)
        p_hash = compute_perceptual_hash(query_img)
        
        if options.debug_mode:
            logger.debug(f"Query image hashes - avgHash: {avg_hash}, pHash: {p_hash}")
        
        # Create a more complex query based on the query type
        cursor = db_conn.cursor()
        
        if is_jpg_query:
            # For JPG queries, boost the chance of finding related RAW files
            base_filename = get_base_filename(options.query_path)
            search_pattern = f"%{base_filename}%"
            
            if options.debug_mode:
                logger.debug(f"Using filename pattern search for JPG query: {search_pattern}")
            
            if options.source_prefix:
                cursor.execute(
                    "SELECT path, source_prefix, average_hash, perceptual_hash, format FROM images "
                    "WHERE (source_prefix = ? OR 1=0) AND (path LIKE ? OR 1=1)",
                    (options.source_prefix, search_pattern)
                )
            else:
                cursor.execute(
                    "SELECT path, source_prefix, average_hash, perceptual_hash, format FROM images "
                    "WHERE path LIKE ? OR 1=1",
                    (search_pattern,)
                )
        else:
            # Standard query for other file types
            if options.source_prefix:
                cursor.execute(
                    "SELECT path, source_prefix, average_hash, perceptual_hash, format FROM images "
                    "WHERE source_prefix = ?",
                    (options.source_prefix,)
                )
            else:
                cursor.execute("SELECT path, source_prefix, average_hash, perceptual_hash, format FROM images")
        
        rows = cursor.fetchall()
        
        matches = []
        processed = 0
        raw_processed = 0
        start_time = time.time()
        
        # Process matches with ThreadPoolExecutor for parallel execution
        mutex = Lock()
        
        def process_candidate(row):
            nonlocal raw_processed
            
            path, source_prefix, db_avg_hash, db_p_hash, format_str = row
            
            # Check if file still exists
            if not os.path.exists(path):
                return None
            
            # Check if the candidate is a RAW file
            is_raw_candidate = is_raw_format(path)
            if is_raw_candidate:
                with mutex:
                    raw_processed += 1
            
            # Calculate hamming distance
            avg_hash_distance = calculate_hamming_distance(avg_hash, db_avg_hash)
            p_hash_distance = calculate_hamming_distance(p_hash, db_p_hash)
            
            # Determine thresholds based on file types
            if (is_raw_query and is_jpg_format(path)) or (is_jpg_query and is_raw_candidate):
                avg_threshold = 20    # Much more lenient
                p_hash_threshold = 25  # Much more lenient
                
                if options.debug_mode:
                    logger.debug(f"Using very lenient thresholds for RAW-JPG comparison between "
                                f"{options.query_path} and {path}")
            elif is_raw_query or is_raw_candidate:
                # Somewhat lenient for other RAW-involved comparisons
                avg_threshold = 15
                p_hash_threshold = 18
            else:
                # Standard thresholds for normal image comparisons
                avg_threshold = 10
                p_hash_threshold = 12
            
            # Special handling for filename-based matching
            if is_jpg_query and is_raw_candidate:
                # Check if the filenames suggest they're related
                if are_filenames_related(options.query_path, path):
                    if options.debug_mode:
                        logger.debug(f"Filename relationship detected between {options.query_path} and {path}, "
                                    f"forcing comparison")
                    
                    # For related filenames, force SSIM comparison regardless of hash distance
                    avg_threshold = 64  # Essentially bypassing the check (8x8 hash has max 64 bits)
                    p_hash_threshold = 64  # Essentially bypassing the check
            
            # If hash distance is within threshold, compute SSIM for more accurate comparison
            if avg_hash_distance <= avg_threshold or p_hash_distance <= p_hash_threshold:
                if options.debug_mode:
                    if is_raw_candidate or is_raw_query:
                        logger.debug(f"RAW image potential match found: {path} "
                                    f"(avgHashDist: {avg_hash_distance}/{avg_threshold}, "
                                    f"pHashDist: {p_hash_distance}/{p_hash_threshold})")
                    else:
                        logger.debug(f"Potential match found: {path} "
                                    f"(avgHashDist: {avg_hash_distance}/{avg_threshold}, "
                                    f"pHashDist: {p_hash_distance}/{p_hash_threshold})")
                
                # Load candidate image
                try:
                    candidate_img = None
                    for loader in registry.get_loaders():
                        if loader.can_load(path):
                            candidate_img = loader.load_image(path)
                            break
                    
                    if candidate_img is None:
                        return None
                    
                    # Adjust threshold for RAW-JPG comparisons
                    local_threshold = options.threshold
                    
                    # Use a more lenient SSIM threshold for RAW-JPG comparisons
                    if (is_raw_query and is_jpg_format(path)) or (is_jpg_query and is_raw_format(path)):
                        # Lower the threshold by 20% for RAW-JPG comparisons
                        local_threshold = options.threshold * 0.8
                        
                        if options.debug_mode:
                            logger.debug(f"Using reduced SSIM threshold of {local_threshold:.2f} "
                                        f"for RAW-JPG comparison with {path}")
                    
                    ssim_score = compute_ssim(query_img, candidate_img)
                    
                    # If SSIM score is above threshold, add to matches
                    if ssim_score >= local_threshold:
                        if options.debug_mode:
                            logger.debug(f"Match confirmed: {path} (SSIM: {ssim_score:.4f} >= {local_threshold:.4f})")
                        
                        return ImageMatch(
                            path=path,
                            source_prefix=source_prefix,
                            ssim_score=ssim_score
                        )
                    elif options.debug_mode and (is_raw_query or is_raw_format(path)):
                        logger.debug(f"RAW image match rejected: {path} "
                                    f"(SSIM: {ssim_score:.4f} < {local_threshold:.4f})")
                
                except Exception as e:
                    if options.debug_mode:
                        logger.warning(f"Failed to load candidate image {path}: {e}")
            
            return None
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            for i, result in enumerate(executor.map(process_candidate, rows)):
                processed += 1
                
                if result is not None:
                    matches.append(result)
                
                # Log progress every 100 images in debug mode
                if options.debug_mode and processed % 100 == 0:
                    elapsed = time.time() - start_time
                    logger.debug(f"Search progress: {processed} images processed ({raw_processed} RAW) "
                                f"in {elapsed:.2f}s")
        
        if options.debug_mode:
            logger.debug(f"Search completed. Total images processed: {processed} ({raw_processed} RAW), "
                        f"Matches found: {len(matches)}")
        
        # Sort matches by SSIM score (higher is better)
        matches.sort(key=lambda x: x.ssim_score, reverse=True)
        
        return matches
    
    except Exception as e:
        logger.error(f"Error in find_similar_images: {e}")
        return []