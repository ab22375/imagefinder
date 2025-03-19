# rust_test.py - Place this in your src/imagefinder directory
try:
    # Import directly - the Rust module is installed at the Python level, not inside your package
    import raw_processor
    print("Rust module successfully loaded!")
    
    # Test a simple function if available
    try:
        # This will only work if you have an 8x8 image to test with
        import numpy as np
        test_img = np.zeros((8, 8), dtype=np.uint8)
        hash_result = raw_processor.rust_compute_average_hash(test_img)
        print(f"Test hash result: {hash_result}")
    except Exception as e:
        print(f"Error testing function: {e}")
        
except ImportError as e:
    print(f"Failed to import raw_processor: {e}")