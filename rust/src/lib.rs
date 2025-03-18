// src/lib.rs
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyByteArray;
use pyo3::exceptions::PyIOError;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use numpy::{PyArray2, IntoPyArray, PyReadonlyArray2};
use std::fs;

// Raw processing libraries
use rawloader::{decode_file, RawImageData};
use image::{ImageBuffer, Rgb};

#[pyfunction]
fn rust_convert_raw_to_jpg(raw_path: &str, jpg_path: &str) -> PyResult<bool> {
    // Attempt to decode the RAW file
    let raw_image = match decode_file(raw_path) {
        Ok(img) => img,
        Err(e) => {
            return Err(PyIOError::new_err(
                format!("Failed to decode RAW file: {}", e)
            ));
        }
    };

    // Process the image based on its data type
    let result = match raw_image.data {
        RawImageData::Integer(ref data) => {
            // Handle integer data
            process_and_save_integer(data, &raw_image, jpg_path)
        },
        RawImageData::Float(ref data) => {
            // Handle float data
            process_and_save_float(data, &raw_image, jpg_path)
        }
    };

    // Check if image was saved successfully
    match result {
        Ok(_) => Ok(true),
        Err(e) => Err(PyIOError::new_err(format!("Failed to save JPG: {}", e)))
    }
}

// Process integer data and save as JPG
fn process_and_save_integer(
    data: &Vec<u16>, 
    raw_image: &rawloader::RawImage, 
    jpg_path: &str
) -> Result<(), Box<dyn std::error::Error>> {
    let width = raw_image.width;
    let height = raw_image.height;
    
    // Create a new RGB image buffer
    let mut img_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width as u32, height as u32);
    
    // Simple demosaicing and conversion from raw data to RGB
    // Note: This is a simplified approach - real demosaicing is more complex
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if idx < data.len() {
                // Convert 16-bit to 8-bit (simple tone mapping)
                let pixel_value = (data[idx] >> 8) as u8;
                
                // In a real implementation, you'd perform proper demosaicing
                // This is a grayscale approximation
                img_buffer.put_pixel(x as u32, y as u32, Rgb([pixel_value, pixel_value, pixel_value]));
            }
        }
    }
    
    // Save as JPEG
    img_buffer.save(jpg_path)?;
    Ok(())
}

// Process float data and save as JPG
fn process_and_save_float(
    data: &Vec<f32>, 
    raw_image: &rawloader::RawImage, 
    jpg_path: &str
) -> Result<(), Box<dyn std::error::Error>> {
    let width = raw_image.width;
    let height = raw_image.height;
    
    // Create a new RGB image buffer
    let mut img_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width as u32, height as u32);
    
    // Simple processing
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if idx < data.len() {
                // Convert float (0.0-1.0) to 8-bit (simple tone mapping)
                let pixel_value = (data[idx].max(0.0).min(1.0) * 255.0) as u8;
                
                // In a real implementation, you'd perform proper demosaicing
                // This is a grayscale approximation
                img_buffer.put_pixel(x as u32, y as u32, Rgb([pixel_value, pixel_value, pixel_value]));
            }
        }
    }
    
    // Save as JPEG
    img_buffer.save(jpg_path)?;
    Ok(())
}

#[pyfunction]
fn rust_raw_to_grayscale(py: Python<'_>, path: &str) -> PyResult<Py<PyArray2<u8>>> {
    // Attempt to decode the RAW file
    let raw_image = match decode_file(path) {
        Ok(img) => img,
        Err(e) => {
            return Err(PyIOError::new_err(
                format!("Failed to decode RAW file: {}", e)
            ));
        }
    };
    
    let width = raw_image.width;
    let height = raw_image.height;
    
    // Create a buffer for our grayscale image
    let mut grayscale = vec![0u8; width * height];
    
    // Convert raw data to grayscale based on its data type
    match raw_image.data {
        RawImageData::Integer(ref data) => {
            for y in 0..height {
                for x in 0..width {
                    let idx = y * width + x;
                    if idx < data.len() {
                        // Convert 16-bit to 8-bit
                        grayscale[idx] = (data[idx] >> 8) as u8;
                    }
                }
            }
        },
        RawImageData::Float(ref data) => {
            for y in 0..height {
                for x in 0..width {
                    let idx = y * width + x;
                    if idx < data.len() {
                        // Convert float (0.0-1.0) to 8-bit
                        grayscale[idx] = (data[idx].max(0.0).min(1.0) * 255.0) as u8;
                    }
                }
            }
        }
    }
    
    // Create a numpy array from our grayscale data
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
}

#[pyfunction]
fn rust_compute_average_hash(py: Python<'_>, image: PyReadonlyArray2<u8>) -> PyResult<String> {
    let arr = image.as_array();
    if arr.shape()[0] != 8 || arr.shape()[1] != 8 {
        return Err(PyIOError::new_err("Image must be 8x8 for average hash"));
    }
    
    // Calculate the average pixel value
    let mut sum = 0u32;
    for i in 0..8 {
        for j in 0..8 {
            sum += arr[[i, j]] as u32;
        }
    }
    let avg = sum / 64;
    
    // Compute the hash
    let mut hash = String::with_capacity(64);
    for i in 0..8 {
        for j in 0..8 {
            if arr[[i, j]] as u32 >= avg {
                hash.push('1');
            } else {
                hash.push('0');
            }
        }
    }
    
    Ok(hash)
}

#[pyfunction]
fn rust_compute_perceptual_hash(py: Python<'_>, image: PyReadonlyArray2<u8>) -> PyResult<String> {
    let arr = image.as_array();
    if arr.shape()[0] != 32 || arr.shape()[1] != 32 {
        return Err(PyIOError::new_err("Image must be 32x32 for perceptual hash"));
    }
    
    const REGIONS: usize = 8;
    let region_height = arr.shape()[0] / REGIONS;
    let region_width = arr.shape()[1] / REGIONS;
    
    // Calculate region values
    let mut region_values = Vec::with_capacity(REGIONS * REGIONS);
    
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
            
            region_values.push(sum as f32 / count as f32);
        }
    }
    
    // Calculate median
    let mut sorted_values = region_values.clone();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if sorted_values.len() % 2 == 0 {
        (sorted_values[sorted_values.len() / 2 - 1] + sorted_values[sorted_values.len() / 2]) / 2.0
    } else {
        sorted_values[sorted_values.len() / 2]
    };
    
    // Create hash
    let mut hash = String::with_capacity(64);
    for val in region_values {
        if val > median {
            hash.push('1');
        } else {
            hash.push('0');
        }
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
    Ok(())
}