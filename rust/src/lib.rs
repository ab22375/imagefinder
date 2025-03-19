// src/lib.rs
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::exceptions::PyIOError;
use std::path::Path;
use std::process::Command;
use numpy::{PyArray2, PyReadonlyArray2};
use std::io::Write;
use std::fs::File;
use std::time::{Duration, Instant};

// Raw processing libraries
use rawloader::{decode_file, RawImageData};
use image::{ImageBuffer, Rgb, DynamicImage, GenericImageView, imageops};

// Constants for optimization
const THUMBNAIL_SIZE: u32 = 512; // Size for thumbnails used in hashing
const TIMEOUT_SECONDS: u64 = 4; // Timeout for external tools

/// Check if a file is a specific RAW format
#[pyfunction]
fn is_specific_raw_format(path: &str, format: &str) -> bool {
    Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map_or(false, |ext| ext.to_lowercase() == format.to_lowercase())
}

/// Special function for RAF files optimized for speed
#[pyfunction]
fn rust_process_raf_file(path: &str, jpg_path: &str) -> PyResult<bool> {
    // Start a timer for performance tracking
    let start = Instant::now();
    
    // RAF files need special handling - try several approaches in parallel
    // First, try to extract embedded JPEG preview with exiftool (fastest)
    let result = extract_preview_with_exiftool(path, jpg_path);
    if result {
        return Ok(true);
    }
    
    // Check if timing out
    if start.elapsed() > Duration::from_secs(TIMEOUT_SECONDS) {
        return Err(PyIOError::new_err("RAF processing timeout"));
    }
    
    // If exiftool failed, try dcraw with simplified options
    let result = extract_with_dcraw_simple(path, jpg_path);
    if result {
        return Ok(true);
    }
    
    // Check if timing out
    if start.elapsed() > Duration::from_secs(TIMEOUT_SECONDS) {
        return Err(PyIOError::new_err("RAF processing timeout"));
    }
    
    // Last resort: try using libraw via dcraw_emu with specific options for Fuji
    let result = extract_with_libraw_fuji(path, jpg_path);
    if result {
        return Ok(true);
    }
    
    Err(PyIOError::new_err("Failed to process RAF file with any available method"))
}

/// Extract preview image using exiftool (fastest method)
/// Extract preview image using exiftool (fastest method)
fn extract_preview_with_exiftool(path: &str, jpg_path: &str) -> bool {
    // Try different preview types in order of preference
    let preview_tags = [
        "-PreviewImage",
        "-JpgFromRaw", 
        "-ThumbnailImage",
        "-OtherImage",
        "-EmbeddedImage"
    ];
    
    for tag in &preview_tags {
        let exiftool_result = Command::new("exiftool")
            .args(&["-b", tag, "-w", jpg_path, path])
            .output();
        
        if let Ok(output) = exiftool_result {
            if output.status.success() && Path::new(jpg_path).exists() {
                // Check file size to ensure its a valid image
                if let Ok(metadata) = std::fs::metadata(jpg_path) {
                    if metadata.len() > 10000 { // More than 10KB is likely a valid image
                        return true;
                    }
                }
            }
        }
    }
    
    false
}
/// Extract with dcraw using minimal processing options (faster)
fn extract_with_dcraw_simple(path: &str, jpg_path: &str) -> bool {
    // Extract embedded thumbnail (very fast)
    let dcraw_thumb_result = Command::new("dcraw")
        .args(&["-e", path])
        .output();
    
    if let Ok(output) = dcraw_thumb_result {
        if output.status.success() {
            // Check if thumb_*.jpg was created
            let path_obj = Path::new(path);
            let filename = path_obj.file_name().unwrap_or_default().to_str().unwrap_or("");
            let thumb_path = path_obj.with_file_name(format!("thumb_{}", filename)).with_extension("jpg");
            
            if thumb_path.exists() {
                if let Ok(_) = std::fs::copy(thumb_path, jpg_path) {
                    let _ = std::fs::remove_file(thumb_path); // Clean up
                    return true;
                }
            }
        }
    }
    
    // If thumbnail extraction failed, try quick conversion
    let dcraw_result = Command::new("dcraw")
        .args(&["-c", "-h", "-q", "0", path]) // -h = half-size, -q 0 = fast interpolation
        .output();
    
    if let Ok(output) = dcraw_result {
        if output.status.success() {
            // Save output to a temporary PPM file
            let temp_ppm = format!("{}.ppm", jpg_path);
            if let Ok(mut file) = File::create(&temp_ppm) {
                if file.write_all(&output.stdout).is_ok() {
                    // Convert PPM to JPG
                    if let Ok(img) = image::open(&temp_ppm) {
                        if img.save(jpg_path).is_ok() {
                            let _ = std::fs::remove_file(&temp_ppm); // Clean up
                            return true;
                        }
                    }
                }
                let _ = std::fs::remove_file(&temp_ppm); // Clean up on failure
            }
        }
    }
    
    false
}

/// Extract with libraw using Fuji-specific options
/// Extract with libraw using Fuji-specific options
fn extract_with_libraw_fuji(path: &str, jpg_path: &str) -> bool {
    // First try with dcraw_emu to extract embedded preview (fastest method)
    let dcraw_emu_result = Command::new("dcraw_emu")
        .args(&["-e", path]) // Extract embedded preview
        .output();
    
    if let Ok(output) = dcraw_emu_result {
        if output.status.success() {
            // Check if thumb_*.jpg was created
            let path_obj = Path::new(path);
            let filename = path_obj.file_name().unwrap_or_default().to_str().unwrap_or("");
            let thumb_path = path_obj.with_file_name(format!("thumb_{}", filename)).with_extension("jpg");
            
            if thumb_path.exists() {
                if let Ok(metadata) = std::fs::metadata(&thumb_path) {
                    // Make sure the extracted preview is not too small
                    if metadata.len() > 10000 { // Minimum size check (10KB)
                        if let Ok(_) = std::fs::copy(thumb_path, jpg_path) {
                            let _ = std::fs::remove_file(thumb_path); // Clean up
                            return true;
                        }
                    }
                }
                let _ = std::fs::remove_file(thumb_path); // Clean up if too small
            }
        }
    }
    
    // Try additional embedded preview extraction with exiftool
    let exiftool_result = Command::new("exiftool")
        .args(&["-b", "-JpgFromRaw", "-w", jpg_path, path])
        .output();
    
    if let Ok(output) = exiftool_result {
        if output.status.success() && Path::new(jpg_path).exists() {
            if let Ok(metadata) = std::fs::metadata(jpg_path) {
                if metadata.len() > 10000 { // More than 10KB is likely a valid image
                    return true;
                }
            }
        }
    }
    
    // If preview extraction failed, try fast conversion with -M flag for speed
    let dcraw_emu_fast_result = Command::new("dcraw_emu")
        .args(&["-c", "-M", "-h", "-q", "0", "-fbdd", "1", "-o", "0", path])
        // -M = use quick interpolation, -h = half-size, -q 0 = fast quality
        // -fbdd 1 = fixed pattern noise reduction, -o 0 = raw color
        .output();
    
    if let Ok(output) = dcraw_emu_fast_result {
        if output.status.success() {
            // Save output to a temporary PPM file
            let temp_ppm = format!("{}.ppm", jpg_path);
            if let Ok(mut file) = File::create(&temp_ppm) {
                if file.write_all(&output.stdout).is_ok() {
                    // Convert PPM to JPG
                    if let Ok(img) = image::open(&temp_ppm) {
                        if img.save(jpg_path).is_ok() {
                            let _ = std::fs::remove_file(&temp_ppm); // Clean up
                            return true;
                        }
                    }
                }
                let _ = std::fs::remove_file(&temp_ppm); // Clean up on failure
            }
        }
    }
    
    // Last resort: Try with specific Fuji X-Trans settings (slower)
    let dcraw_emu_xtrans_result = Command::new("dcraw_emu")
        .args(&["-M", "-q", "0", "-h", "-f", "-fbdd", "1", path])
        // -M = quick interpolation, -q 0 = fast, -h = half-size
        // -f = Fuji xtrans mode, -fbdd 1 = fixed pattern noise reduction
        .output();
    
    if let Ok(output) = dcraw_emu_xtrans_result {
        if output.status.success() {
            // Save output to a temporary PPM file
            let temp_ppm = format!("{}.ppm", jpg_path);
            if let Ok(mut file) = File::create(&temp_ppm) {
                if file.write_all(&output.stdout).is_ok() {
                    // Convert PPM to JPG
                    if let Ok(img) = image::open(&temp_ppm) {
                        if img.save(jpg_path).is_ok() {
                            let _ = std::fs::remove_file(&temp_ppm); // Clean up
                            return true;
                        }
                    }
                }
                let _ = std::fs::remove_file(&temp_ppm); // Clean up on failure
            }
        }
    }
    
    false
}

/// Convert a RAW image to a processed RGB image with performance optimizations
#[pyfunction]
fn rust_convert_raw_to_jpg(path: &str, jpg_path: &str) -> PyResult<bool> {
    // Check if its a Fuji RAF file - use dedicated function
    if is_specific_raw_format(path, "raf") {
        return rust_process_raf_file(path, jpg_path);
    }
    
    // Start a timer for performance tracking
    let start = Instant::now();
    
    // Get file extension to identify the RAW format
    let ext = Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();
    
    // For each format type, try the fastest method first
    
    // Try extracting embedded preview first (fastest method for all formats)
    if try_extract_embedded_preview(path, jpg_path) {
        return Ok(true);
    }
    
    // If timing out, bail early
    if start.elapsed() > Duration::from_secs(TIMEOUT_SECONDS) {
        return Err(PyIOError::new_err("RAW processing timeout"));
    }
    
    // Try specific optimizations based on format
    match ext.as_str() {
        "arw" => {
            // Sony ARW specific processing
            if try_sony_arw_processing(path, jpg_path) {
                return Ok(true);
            }
        },
        "cr2" | "cr3" => {
            // Canon specific processing
            if try_canon_cr_processing(path, jpg_path) {
                return Ok(true);
            }
        },
        "nef" => {
            // Nikon specific processing
            if try_nikon_nef_processing(path, jpg_path) {
                return Ok(true);
            }
        },
        _ => {
            // Try rawloader for general formats (works well with DNG)
            if try_rawloader_processing(path, jpg_path) {
                return Ok(true);
            }
        }
    }
    
    // If timing out, bail early
    if start.elapsed() > Duration::from_secs(TIMEOUT_SECONDS) {
        return Err(PyIOError::new_err("RAW processing timeout"));
    }
    
    // Generic fallback processing
    if try_generic_raw_processing(path, jpg_path) {
        return Ok(true);
    }
    
    Err(PyIOError::new_err(format!("Failed to process RAW file: {}", path)))
}

/// Try to extract embedded preview (fastest method)
fn try_extract_embedded_preview(path: &str, jpg_path: &str) -> bool {
    // Try exiftool first (it is usually fastest)
    if extract_preview_with_exiftool(path, jpg_path) {
        return true;
    }
    
    // Try dcraw preview extraction
    let dcraw_thumb_result = Command::new("dcraw")
        .args(&["-e", path])
        .output();
    
    if let Ok(output) = dcraw_thumb_result {
        if output.status.success() {
            // Check if thumb_*.jpg was created
            let path_obj = Path::new(path);
            let filename = path_obj.file_name().unwrap_or_default().to_str().unwrap_or("");
            let thumb_path = path_obj.with_file_name(format!("thumb_{}", filename)).with_extension("jpg");
            
            if thumb_path.exists() {
                if let Ok(_) = std::fs::copy(thumb_path, jpg_path) {
                    let _ = std::fs::remove_file(thumb_path); // Clean up
                    return true;
                }
            }
        }
    }
    
    false
}

/// Sony ARW specific processing
fn try_sony_arw_processing(path: &str, jpg_path: &str) -> bool {
    // Sony ARW works well with custom dcraw settings
    let dcraw_sony_result = Command::new("dcraw")
        .args(&["-c", "-w", "-h", "-q", "0", "-o", "0", path]) 
        // -h = half size, -q 0 = fast quality, -o 0 = raw color
        .output();
    
    if let Ok(output) = dcraw_sony_result {
        if output.status.success() {
            // Save output to a temporary PPM file
            let temp_ppm = format!("{}.ppm", jpg_path);
            if let Ok(mut file) = File::create(&temp_ppm) {
                if file.write_all(&output.stdout).is_ok() {
                    // Convert PPM to JPG
                    if let Ok(img) = image::open(&temp_ppm) {
                        if img.save(jpg_path).is_ok() {
                            let _ = std::fs::remove_file(&temp_ppm); // Clean up
                            return true;
                        }
                    }
                }
                let _ = std::fs::remove_file(&temp_ppm); // Clean up on failure
            }
        }
    }
    
    false
}

/// Canon CR2/CR3 specific processing
fn try_canon_cr_processing(path: &str, jpg_path: &str) -> bool {
    // Canon works well with these dcraw settings
    let dcraw_canon_result = Command::new("dcraw")
        .args(&["-c", "-w", "-h", "-q", "0", path]) 
        // -h = half size (faster), -q 0 = fast quality
        .output();
    
    if let Ok(output) = dcraw_canon_result {
        if output.status.success() {
            // Save output to a temporary PPM file
            let temp_ppm = format!("{}.ppm", jpg_path);
            if let Ok(mut file) = File::create(&temp_ppm) {
                if file.write_all(&output.stdout).is_ok() {
                    // Convert PPM to JPG
                    if let Ok(img) = image::open(&temp_ppm) {
                        if img.save(jpg_path).is_ok() {
                            let _ = std::fs::remove_file(&temp_ppm); // Clean up
                            return true;
                        }
                    }
                }
                let _ = std::fs::remove_file(&temp_ppm); // Clean up on failure
            }
        }
    }
    
    false
}

/// Nikon NEF specific processing
fn try_nikon_nef_processing(path: &str, jpg_path: &str) -> bool {
    // Nikon specific settings
    let dcraw_nikon_result = Command::new("dcraw")
        .args(&["-c", "-w", "-h", "-q", "0", "-o", "1", path]) 
        // -h = half size, -q 0 = fast, -o 1 = sRGB (better for Nikon)
        .output();
    
    if let Ok(output) = dcraw_nikon_result {
        if output.status.success() {
            // Save output to a temporary PPM file
            let temp_ppm = format!("{}.ppm", jpg_path);
            if let Ok(mut file) = File::create(&temp_ppm) {
                if file.write_all(&output.stdout).is_ok() {
                    // Convert PPM to JPG
                    if let Ok(img) = image::open(&temp_ppm) {
                        if img.save(jpg_path).is_ok() {
                            let _ = std::fs::remove_file(&temp_ppm); // Clean up
                            return true;
                        }
                    }
                }
                let _ = std::fs::remove_file(&temp_ppm); // Clean up on failure
            }
        }
    }
    
    false
}

/// Try processing with rawloader (works well for DNG)
fn try_rawloader_processing(path: &str, jpg_path: &str) -> bool {
    match decode_file(path) {
        Ok(raw_image) => {
            // Process the image based on its data type
            match process_and_save_image(&raw_image, jpg_path) {
                Ok(_) => true,
                Err(_) => false
            }
        },
        Err(_) => false
    }
}

/// Generic RAW processing fallback
fn try_generic_raw_processing(path: &str, jpg_path: &str) -> bool {
    // Try dcraw with generic options
    let dcraw_result = Command::new("dcraw")
        .args(&["-c", "-w", "-h", "-q", "0", path]) // Use fast options
        .output();
    
    if let Ok(output) = dcraw_result {
        if output.status.success() {
            // Save output to a temporary PPM file
            let temp_ppm = format!("{}.ppm", jpg_path);
            if let Ok(mut file) = File::create(&temp_ppm) {
                if file.write_all(&output.stdout).is_ok() {
                    // Convert PPM to JPG
                    if let Ok(img) = image::open(&temp_ppm) {
                        if img.save(jpg_path).is_ok() {
                            let _ = std::fs::remove_file(&temp_ppm); // Clean up
                            return true;
                        }
                    }
                }
                let _ = std::fs::remove_file(&temp_ppm); // Clean up on failure
            }
        }
    }
    
    // Last resort: Try dcraw_emu
    let dcraw_emu_result = Command::new("dcraw_emu")
        .args(&["-T", "-h", "-q", "0", path]) // Use fast options
        .output();
    
    if let Ok(output) = dcraw_emu_result {
        if output.status.success() {
            // Save output to a temporary TIFF file
            let temp_tiff = format!("{}.tiff", jpg_path);
            if let Ok(mut file) = File::create(&temp_tiff) {
                if file.write_all(&output.stdout).is_ok() {
                    // Convert TIFF to JPG
                    if let Ok(img) = image::open(&temp_tiff) {
                        if img.save(jpg_path).is_ok() {
                            let _ = std::fs::remove_file(&temp_tiff); // Clean up
                            return true;
                        }
                    }
                }
                let _ = std::fs::remove_file(&temp_tiff); // Clean up on failure
            }
        }
    }
    
    false
}

/// Process raw image data and save as JPG with improved processing
fn process_and_save_image(raw_image: &rawloader::RawImage, jpg_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let width = raw_image.width;
    let height = raw_image.height;
    
    // Create a new RGB image buffer
    let mut img_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width as u32, height as u32);
    
    // Apply simple debayering (this is rudimentary and could be improved)
    match &raw_image.data {
        RawImageData::Integer(data) => {
            // Create a parallel iterator for better performance
            // In real production code, you'd use rayon for this, but we'll keep it simple
            for y in 0..height {
                for x in 0..width {
                    let idx = y * width + x;
                    if idx < data.len() {
                        // Simple conversion from 16-bit to 8-bit with gamma correction
                        let value = ((data[idx] as f32 / 65535.0).powf(0.45) * 255.0) as u8;
                        
                        // Simple color estimation based on Bayer pattern (RGGB assumed)
                        let pattern_idx = (y % 2) * 2 + (x % 2);
                        let (r, g, b) = match pattern_idx {
                            0 => (value, value/2, value/2), // R
                            1 | 2 => (value/2, value, value/2), // G
                            _ => (value/2, value/2, value),  // B
                        };
                        
                        img_buffer.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
                    }
                }
            }
        },
        RawImageData::Float(data) => {
            for y in 0..height {
                for x in 0..width {
                    let idx = y * width + x;
                    if idx < data.len() {
                        // Convert float to 8-bit with gamma correction
                        let value = ((data[idx].max(0.0).min(1.0)).powf(0.45) * 255.0) as u8;
                        
                        // Simple color estimation
                        let pattern_idx = (y % 2) * 2 + (x % 2);
                        let (r, g, b) = match pattern_idx {
                            0 => (value, value/2, value/2), // R
                            1 | 2 => (value/2, value, value/2), // G
                            _ => (value/2, value/2, value),  // B
                        };
                        
                        img_buffer.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
                    }
                }
            }
        }
    }
    
    // Convert to DynamicImage and resize to a reasonable size if very large
    let mut img = DynamicImage::ImageRgb8(img_buffer);
    
    // Resize if image is very large (helps with performance and quality)
    if width > 2000 || height > 2000 {
        img = img.resize(width as u32 / 2, height as u32 / 2, imageops::FilterType::Triangle);
    }
    
    // Save as JPEG with moderate quality (85%)
    img.save_with_format(jpg_path, image::ImageFormat::Jpeg)?;
    
    Ok(())
}

/// Convert RAW directly to grayscale for hashing (optimized version)
#[pyfunction]
fn rust_raw_to_grayscale(py: Python<'_>, path: &str) -> PyResult<Py<PyArray2<u8>>> {
    // First try to convert to JPG
    let temp_jpg = format!("{}.temp.jpg", path);
    
    let result = if is_specific_raw_format(path, "raf") {
        rust_process_raf_file(path, &temp_jpg)
    } else {
        rust_convert_raw_to_jpg(path, &temp_jpg)
    };
    
    match result {
        Ok(_) => {
            // Process the temporary JPG to grayscale
            match image::open(&temp_jpg) {
                Ok(img) => {
                    // Convert to grayscale
                    let gray_img = img.grayscale();
                    
                    // Resize to thumbnail size for hashing
                    let resized = gray_img.resize_exact(
                        THUMBNAIL_SIZE, 
                        THUMBNAIL_SIZE, 
                        imageops::FilterType::Triangle
                    );
                    
                    // Convert to numpy array
                    let height = resized.height() as usize;
                    let width = resized.width() as usize;
                    let mut grayscale = vec![0u8; width * height];
                    
                    for y in 0..height {
                        for x in 0..width {
                            let pixel = resized.get_pixel(x as u32, y as u32);
                            grayscale[y * width + x] = pixel[0]; // Take first channel
                        }
                    }
                    
                    // Clean up temp file
                    let _ = std::fs::remove_file(&temp_jpg);
                    
                    // Create numpy array
                    unsafe {
                        let buffer = numpy::PyArray2::<u8>::new(
                            py, 
                            [height, width], 
                            false
                        );
                        
                        let dataptr = buffer.as_array_mut().as_mut_ptr();
                        std::ptr::copy_nonoverlapping(
                            grayscale.as_ptr(), 
                            dataptr, 
                            width * height
                        );
                        
                        Ok(buffer.into())
                    }
                },
                Err(e) => {
                    let _ = std::fs::remove_file(&temp_jpg); // Clean up
                    Err(PyIOError::new_err(format!("Failed to open converted image: {}", e)))
                }
            }
        },
        Err(e) => {
            let _ = std::fs::remove_file(&temp_jpg); // Clean up if it exists
            Err(e)
        }
    }
}

// Optimized hash functions
#[pyfunction]
fn rust_compute_average_hash(_py: Python<'_>, image: PyReadonlyArray2<u8>) -> PyResult<String> {
    let arr = image.as_array();
    if arr.shape()[0] != 8 || arr.shape()[1] != 8 {
        return Err(PyIOError::new_err("Image must be 8x8 for average hash"));
    }
    
    // Calculate the average pixel value (optimized)
    let mut sum = 0u32;
    for row in arr.rows() {
        for &pixel in row {
            sum += pixel as u32;
        }
    }
    let avg = sum / 64;
    
    // Compute the hash (bit-packed for efficiency)
    let mut hash = String::with_capacity(64);
    
    for row in arr.rows() {
        for &pixel in row {
            if pixel as u32 >= avg {
                hash.push('1');
            } else {
                hash.push('0');
            }
        }
    }
    
    Ok(hash)
}

#[pyfunction]
fn rust_compute_perceptual_hash(_py: Python<'_>, image: PyReadonlyArray2<u8>) -> PyResult<String> {
    let arr = image.as_array();
    if arr.shape()[0] != 32 || arr.shape()[1] != 32 {
        return Err(PyIOError::new_err("Image must be 32x32 for perceptual hash"));
    }
    
    const REGIONS: usize = 8;
    let region_height = arr.shape()[0] / REGIONS;
    let region_width = arr.shape()[1] / REGIONS;
    
    // Calculate region values (optimized)
    let mut region_values = vec![0.0; REGIONS * REGIONS];
    
    for i in 0..REGIONS {
        for j in 0..REGIONS {
            let start_y = i * region_height;
            let end_y = (i + 1) * region_height;
            let start_x = j * region_width;
            let end_x = (j + 1) * region_width;
            
            let mut sum = 0u32;
            let mut count = 0u32;
            
            for y in start_y..end_y {
                for x in start_x..end_x {
                    sum += arr[[y, x]] as u32;
                    count += 1;
                }
            }
            
            region_values[i * REGIONS + j] = sum as f32 / count as f32;
        }
    }
    
    // Calculate median (optimized)
    let mut sorted_values = region_values.clone();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted_values[REGIONS * REGIONS / 2];
    
    // Create hash (optimized)
    let mut hash = String::with_capacity(64);
    for val in region_values {
        hash.push(if val > median { '1' } else { '0' });
    }
    
    Ok(hash)
}

/// A Python module implemented in Rust
#[pymodule]
fn raw_processor(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_convert_raw_to_jpg, m)?)?;
    m.add_function(wrap_pyfunction!(rust_raw_to_grayscale, m)?)?;
    m.add_function(wrap_pyfunction!(rust_compute_average_hash, m)?)?;
    m.add_function(wrap_pyfunction!(rust_compute_perceptual_hash, m)?)?;
    m.add_function(wrap_pyfunction!(is_specific_raw_format, m)?)?;
    m.add_function(wrap_pyfunction!(rust_process_raf_file, m)?)?;
    Ok(())
}