#!/usr/bin/env python3
"""
Benchmark script to compare Python vs Rust implementations for RAW processing
with improved error handling and timeout functionality
"""
import os
import time
import argparse
import glob
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import wraps
import threading

TIMEOUTSEC=20

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import functions from scanner.py
from imagefinder.scanner import (
    convert_raw_to_jpg_and_load, 
    compute_average_hash, 
    compute_perceptual_hash,
    is_raw_format
)

# Check if Rust is enabled
try:
    import raw_processor
    RUST_ENABLED = True
    logging.info("Rust module loaded. Will benchmark both implementations.")
except ImportError:
    RUST_ENABLED = False
    logging.warning("Rust module not available. Will only benchmark Python implementation.")

# Create pure Python versions for comparison
import cv2
from imagefinder.scanner import (
    extract_preview_with_exiftool,
    convert_with_dcraw_auto_bright,
    convert_with_dcraw_camera_wb,
    convert_with_rawpy,
    convert_cr3_with_exiftool
)

# Timeout decorator for functions
class TimeoutError(Exception):
    pass

def timeout(seconds, error_message="Function call timed out"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutError(error_message)]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        return wrapper
    return decorator

# Apply timeout to the conversion functions
@timeout(TIMEOUTSEC, f"RAW conversion timed out after {TIMEOUTSEC} seconds")
def timed_py_convert(path):
    """Python-only version of RAW conversion with timeout"""
    # Disable the Rust implementation temporarily
    global RUST_ENABLED
    old_value = RUST_ENABLED
    RUST_ENABLED = False
    
    try:
        # Call the original function with Rust disabled
        return convert_raw_to_jpg_and_load(path)
    finally:
        # Restore the original value
        RUST_ENABLED = old_value

@timeout(TIMEOUTSEC, f"RAW conversion timed out after {TIMEOUTSEC} seconds")
def timed_rust_convert(path):
    """Rust-enabled version of RAW conversion with timeout"""
    return convert_raw_to_jpg_and_load(path)

def py_compute_average_hash(img: np.ndarray) -> str:
    """Python-only version of average hash for benchmarking"""
    # Disable the Rust implementation temporarily
    global RUST_ENABLED
    old_value = RUST_ENABLED
    RUST_ENABLED = False
    
    try:
        # Call the original function with Rust disabled
        return compute_average_hash(img)
    finally:
        # Restore the original value
        RUST_ENABLED = old_value

def py_compute_perceptual_hash(img: np.ndarray) -> str:
    """Python-only version of perceptual hash for benchmarking"""
    # Disable the Rust implementation temporarily
    global RUST_ENABLED
    old_value = RUST_ENABLED
    RUST_ENABLED = False
    
    try:
        # Call the original function with Rust disabled
        return compute_perceptual_hash(img)
    finally:
        # Restore the original value
        RUST_ENABLED = old_value

def benchmark_files(file_paths, output_dir=None, runs=3, timeout_seconds=60):
    """Benchmark RAW processing on multiple files"""
    results = []
    
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for path in file_paths:
        try:
            file_size = os.path.getsize(path) / (1024 * 1024)  # in MB
            file_ext = Path(path).suffix.lower()
            file_name = Path(path).name
            
            logging.info(f"Benchmarking: {file_name} ({file_size:.2f} MB)")
            
            # Skip files that aren't RAW format if we're doing RAW processing
            if not is_raw_format(path):
                logging.info(f"Skipping non-RAW file: {file_name}")
                continue
            
            # Run benchmark for RAW conversion
            benchmark_result = {
                'file': file_name,
                'size_mb': file_size,
                'format': file_ext.lstrip('.'),
            }
            
            # Python timing
            python_times = []
            rust_times = []
            
            # Test Python implementation
            python_success = False
            python_error = None
            python_img = None
            
            for run in range(runs):
                try:
                    start_time = time.time()
                    python_img = timed_py_convert(path)
                    end_time = time.time()
                    
                    if python_img is not None and python_img.size > 0:
                        python_times.append(end_time - start_time)
                        python_success = True
                        
                        # Save output image for visual comparison
                        if output_dir and run == 0:
                            output_path = os.path.join(output_dir, f"python_{file_name}.jpg")
                            cv2.imwrite(output_path, python_img)
                except TimeoutError:
                    python_error = "Timed out after {TIMEOUTSEC} seconds"
                    logging.error(f"Python processing timed out on {file_name}")
                    break  # Don't try additional runs if it times out
                except Exception as e:
                    python_error = str(e)
                    logging.error(f"Python error on {file_name}: {e}")
            
            # Test Rust implementation (if available)
            rust_success = False
            rust_error = None
            rust_img = None
            
            if RUST_ENABLED:
                for run in range(runs):
                    try:
                        start_time = time.time()
                        rust_img = timed_rust_convert(path)  # Uses Rust if available
                        end_time = time.time()
                        
                        if rust_img is not None and rust_img.size > 0:
                            rust_times.append(end_time - start_time)
                            rust_success = True
                            
                            # Save output image for visual comparison
                            if output_dir and run == 0:
                                output_path = os.path.join(output_dir, f"rust_{file_name}.jpg")
                                cv2.imwrite(output_path, rust_img)
                    except TimeoutError:
                        rust_error = "Timed out after {TIMEOUTSEC} seconds"
                        logging.error(f"Rust processing timed out on {file_name}")
                        break  # Don't try additional runs if it times out
                    except Exception as e:
                        rust_error = str(e)
                        logging.error(f"Rust error on {file_name}: {e}")
            
            # Record results for RAW conversion
            python_time = np.mean(python_times) if python_times else None
            rust_time = np.mean(rust_times) if rust_times else None
            
            benchmark_result.update({
                'python_raw_time': python_time,
                'rust_raw_time': rust_time,
                'python_raw_success': python_success,
                'rust_raw_success': rust_success,
                'python_raw_error': python_error,
                'rust_raw_error': rust_error,
            })
            
            # If we have successful image conversion, benchmark hash functions
            if python_img is not None:
                try:
                    # Test average hash
                    python_avg_times = []
                    rust_avg_times = []
                    
                    for run in range(runs):
                        # Python avg hash
                        start_time = time.time()
                        py_compute_average_hash(python_img)
                        python_avg_times.append(time.time() - start_time)
                        
                        # Rust avg hash (if available)
                        if RUST_ENABLED:
                            start_time = time.time()
                            compute_average_hash(python_img)  # Uses Rust if available
                            rust_avg_times.append(time.time() - start_time)
                    
                    # Test perceptual hash
                    python_phash_times = []
                    rust_phash_times = []
                    
                    for run in range(runs):
                        # Python perceptual hash
                        start_time = time.time()
                        py_compute_perceptual_hash(python_img)
                        python_phash_times.append(time.time() - start_time)
                        
                        # Rust perceptual hash (if available)
                        if RUST_ENABLED:
                            start_time = time.time()
                            compute_perceptual_hash(python_img)  # Uses Rust if available
                            rust_phash_times.append(time.time() - start_time)
                    
                    # Record hash timing results
                    benchmark_result.update({
                        'python_avg_hash_time': np.mean(python_avg_times),
                        'rust_avg_hash_time': np.mean(rust_avg_times) if rust_avg_times else None,
                        'python_phash_time': np.mean(python_phash_times),
                        'rust_phash_time': np.mean(rust_phash_times) if rust_phash_times else None,
                    })
                except Exception as e:
                    logging.error(f"Error during hash benchmarking on {file_name}: {e}")
            
            # Calculate speedups
            if python_time and rust_time:
                benchmark_result['raw_speedup'] = python_time / rust_time
            
            if 'python_avg_hash_time' in benchmark_result and 'rust_avg_hash_time' in benchmark_result and benchmark_result['rust_avg_hash_time']:
                benchmark_result['avg_hash_speedup'] = benchmark_result['python_avg_hash_time'] / benchmark_result['rust_avg_hash_time']
            
            if 'python_phash_time' in benchmark_result and 'rust_phash_time' in benchmark_result and benchmark_result['rust_phash_time']:
                benchmark_result['phash_speedup'] = benchmark_result['python_phash_time'] / benchmark_result['rust_phash_time']
            
            # Print results for this file
            if python_success and rust_success:
                logging.info(f"  RAW conversion: Python: {python_time:.3f}s, Rust: {rust_time:.3f}s, Speedup: {benchmark_result.get('raw_speedup', 'N/A'):.2f}x")
                if 'avg_hash_speedup' in benchmark_result:
                    logging.info(f"  Avg Hash: Speedup: {benchmark_result['avg_hash_speedup']:.2f}x")
                if 'phash_speedup' in benchmark_result:
                    logging.info(f"  Perceptual Hash: Speedup: {benchmark_result['phash_speedup']:.2f}x")
            elif python_success:
                logging.info(f"  Python: {python_time:.3f}s, Rust: Failed ({rust_error})")
            elif rust_success:
                logging.info(f"  Python: Failed ({python_error}), Rust: {rust_time:.3f}s")
            else:
                logging.info(f"  Both implementations failed! Python: {python_error}, Rust: {rust_error}")
            
            results.append(benchmark_result)
        except Exception as e:
            logging.error(f"Error benchmarking {os.path.basename(path)}: {e}")
            logging.info("Continuing with next file...")
    
    return pd.DataFrame(results)

def plot_results(df, output_file):
    """Generate plots from benchmark results"""
    plt.figure(figsize=(12, 8))
    
    # Plot RAW conversion time comparison
    raw_data = df[df['python_raw_success'] & df['rust_raw_success']].copy()
    if not raw_data.empty:
        raw_data = raw_data.sort_values('raw_speedup', ascending=False)
        
        plt.subplot(2, 1, 1)
        x = range(len(raw_data))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], raw_data['python_raw_time'], width, label='Python')
        plt.bar([i + width/2 for i in x], raw_data['rust_raw_time'], width, label='Rust')
        
        plt.xlabel('File')
        plt.ylabel('Time (seconds)')
        plt.title('RAW Conversion Time Comparison')
        plt.xticks(x, raw_data['file'], rotation=45, ha='right')
        plt.legend()
        
        # Add speedup annotations
        for i, row in enumerate(raw_data.itertuples()):
            if hasattr(row, 'raw_speedup'):
                plt.text(i, min(row.python_raw_time, row.rust_raw_time)/2, 
                        f"{row.raw_speedup:.1f}x", 
                        ha='center', va='center', 
                        color='white', fontweight='bold')
    
    # Plot hash time comparison
    hash_data = df[df['python_raw_success']].copy()
    if not hash_data.empty and 'avg_hash_speedup' in hash_data.columns:
        hash_data = hash_data.sort_values('avg_hash_speedup', ascending=False)
        
        plt.subplot(2, 1, 2)
        x = range(len(hash_data))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], hash_data['python_avg_hash_time'], width, label='Python Avg Hash')
        plt.bar([i + width/2 for i in x], hash_data['rust_avg_hash_time'], width, label='Rust Avg Hash')
        
        plt.xlabel('File')
        plt.ylabel('Time (seconds)')
        plt.title('Average Hash Time Comparison')
        plt.xticks(x, hash_data['file'], rotation=45, ha='right')
        plt.legend()
        
        # Add speedup annotations
        for i, row in enumerate(hash_data.itertuples()):
            if hasattr(row, 'avg_hash_speedup'):
                plt.text(i, min(row.python_avg_hash_time, row.rust_avg_hash_time)/2, 
                        f"{row.avg_hash_speedup:.1f}x", 
                        ha='center', va='center', 
                        color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark RAW image processing")
    parser.add_argument("--input", "-i", required=True, help="Directory or file(s) to benchmark")
    parser.add_argument("--output", "-o", default="benchmark_results.csv", help="Output CSV file")
    parser.add_argument("--plot", "-p", action="store_true", help="Generate plot of results")
    parser.add_argument("--save-images", "-s", action="store_true", help="Save processed images for comparison")
    parser.add_argument("--runs", "-r", type=int, default=3, help="Number of runs for each benchmark")
    parser.add_argument("--timeout", "-t", type=int, default=60, help="Timeout in seconds for RAW processing")
    args = parser.parse_args()
    
    # Find files to benchmark
    if os.path.isdir(args.input):
        # Find all image files in the directory
        raw_extensions = [".dng", ".raf", ".arw", ".nef", ".cr2", ".cr3", ".nrw", ".srf"]
        files = []
        for ext in raw_extensions:
            files.extend(glob.glob(os.path.join(args.input, f"*{ext}")))
            files.extend(glob.glob(os.path.join(args.input, f"*{ext.upper()}")))
    else:
        # Single file or glob pattern
        files = glob.glob(args.input)
    
    if not files:
        logging.error(f"No files found matching {args.input}")
        return
    
    logging.info(f"Found {len(files)} files to benchmark")
    
    # Create output directory for images if needed
    output_dir = None
    if args.save_images:
        output_dir = os.path.join(os.path.dirname(args.output), "benchmark_images")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    try:
        # Run benchmarks
        results = benchmark_files(files, output_dir, args.runs, args.timeout)
        
        # Save results
        results.to_csv(args.output, index=False)
        logging.info(f"Saved benchmark results to {args.output}")
        
        # Generate plot if requested
        if args.plot and not results.empty:
            plot_file = os.path.splitext(args.output)[0] + ".png"
            plot_results(results, plot_file)
    except KeyboardInterrupt:
        logging.info("Benchmark interrupted by user. Saving partial results...")
        if 'results' in locals() and not results.empty:
            results.to_csv(args.output, index=False)
            logging.info(f"Saved partial results to {args.output}")
            
            if args.plot:
                plot_file = os.path.splitext(args.output)[0] + ".png"
                try:
                    plot_results(results, plot_file)
                except Exception as e:
                    logging.error(f"Error creating plot: {e}")

if __name__ == "__main__":
    main()