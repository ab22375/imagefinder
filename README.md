# ImageFinder

ImageFinder is a tool for scanning, indexing, and finding similar images in large collections. It features specialized support for RAW image formats from various camera manufacturers, making it ideal for photographers.

## Features

- **Scan and index large image collections** with efficient multi-threaded processing
- **Find similar images** using perceptual hashing and SSIM comparison
- **RAW file support** including:
  - Canon (CR2, CR3)
  - Nikon (NEF)
  - Sony (ARW)
  - Fujifilm (RAF)
  - DJI (DNG)
  - And other common RAW formats
- **Customizable similarity thresholds** for fine-tuning search results
- **Source prefix filtering** to organize images from multiple sources
- **SQLite database** for fast indexing and retrieval

## Installation

### From PyPI

```bash
pip install imagefinder
```

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/imagefinder.git
cd imagefinder

# Install with Poetry
poetry install

# Or with pip
pip install .
```

## Dependencies

- Python 3.8+
- OpenCV
- NumPy
- SQLite3
- External tools for RAW processing (optional but recommended):
  - exiftool
  - dcraw
  - rawtherapee-cli

## Usage

### Scanning Images

Scan a folder of images and add them to the database:

```bash
imagefinder scan --folder=/path/to/images --prefix=MyCollection
```

Options:

- `--folder`: Path to the folder containing images (required)
- `--database`: Path to the database file (default: local images.db)
- `--prefix`: Source prefix for organizing collections
- `--force`: Force reindexing of existing images
- `--debug`: Enable detailed logging
- `--logfile`: Specify a custom log file

### Finding Similar Images

Search for images similar to a query image:

```bash
imagefinder search --image=/path/to/query.jpg --threshold=0.85
```

Options:

- `--image`: Path to the query image (required)
- `--database`: Path to the database file
- `--threshold`: Similarity threshold (0.0-1.0, default: 0.8)
- `--prefix`: Filter results by source prefix
- `--debug`: Enable detailed logging

## Examples

### Basic Workflow

1. Scan your image collection:

   ```bash
   imagefinder scan --folder=/Users/me/Pictures
   ```
2. Search for similar images:

   ```bash
   imagefinder search --image=/Users/me/query.jpg
   ```

### Advanced Usage

Scan multiple folders with different source prefixes:

```bash
imagefinder scan --folder=/Users/me/Pictures/Vacation2023 --prefix=Vacation2023
imagefinder scan --folder=/Users/me/Pictures/Family --prefix=Family
```

Search only within a specific collection:

```bash
imagefinder search --image=/Users/me/query.jpg --prefix=Vacation2023
```

Adjust the similarity threshold for more/fewer results:

```bash
imagefinder search --image=/Users/me/query.jpg --threshold=0.7  # More results
imagefinder search --image=/Users/me/query.jpg --threshold=0.9  # Fewer results
```

## How It Works

1. **Scanning**: The tool analyzes images in the specified folder, computing perceptual hashes and storing metadata in an SQLite database.
2. **RAW Processing**: RAW images are handled through specialized loaders that can extract embedded previews or convert to a common format for analysis.
3. **Searching**: When searching, the tool computes the hash of the query image and finds database entries with similar hashes, then performs a structural similarity check (SSIM) for final ranking.

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- This tool was inspired by the need to manage large collections of RAW images
- Special thanks to the creators of OpenCV and the various RAW processing tools
