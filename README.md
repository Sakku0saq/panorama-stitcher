# Panorama Stitcher

A command-line tool that stitches two or more overlapping images into a
single panorama using **SIFT feature detection**, **RANSAC-based homography
estimation**, and **feathered image blending**.

Built as a course project for **CSE3010 – Computer Vision**, applying
concepts from feature extraction, projective geometry, and homography
estimation covered in the syllabus.

---

## Overview

Given a sequence of overlapping photos (e.g. taken while panning a camera
across a scene), this tool automatically:

1. Loads and preprocesses the images
2. Detects SIFT keypoints and matches them between adjacent images
3. Estimates a homography between each pair using RANSAC
4. Warps and blends the images into one panorama

The tool is fully executable from the command line — no GUI required.

## Features

- Stitches **2 or more** images sequentially into a single panorama
- SIFT-based feature detection and Lowe's-ratio-test filtered matching
- RANSAC homography estimation, robust to outlier matches
- Distance-transform feather blending to avoid hard seams
- Automatic image downscaling for performance on large photos
- Structured logging (console + `output/pipeline.log`)
- Clear, actionable error messages for common failure cases (missing
  files, unsupported formats, insufficient overlap, degenerate homography)
- Unit tests covering all four core modules
- Synthetic sample-image generator, so the project runs with zero
  external setup or downloads

## Technologies / Tools Used

- Python 3.10+
- OpenCV (`opencv-python`) — SIFT, feature matching, homography, warping
- NumPy — array/matrix operations
- pytest — unit testing
- `argparse` — CLI interface
- `logging` — structured logging

## Project Structure

```text
panorama-stitcher/
├── main.py                     # CLI entry point
├── requirements.txt
├── README.md
├── statement.md
├── src/
│   ├── __init__.py             # Python package initializer
│   ├── preprocessing.py        # Module 1: image loading & validation
│   ├── feature_detection.py    # Module 2: SIFT detection & matching
│   ├── homography.py           # Module 3: RANSAC homography estimation
│   ├── stitching.py            # Module 4: warping & blending
│   ├── pipeline.py             # Orchestrates modules 1-4
│   └── logger_config.py        # Centralized logging setup
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_feature_detection.py
│   ├── test_homography.py
│   └── test_stitching.py
├── data/
│   ├── generate_samples.py     # Creates synthetic overlapping test images
│   └── sample_images/          # Sample input images
└── output/
    ├── newspaper_panorama.jpg  # Sample generated panorama
    └── pipeline.log            # Pipeline execution log
```

## Setup & Installation

**Prerequisites:** Python 3.10 or higher.

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**

   On Windows:

   ```bash
   venv\Scripts\activate
   ```

   On Linux/macOS:

   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### 1. Generate sample images

The project includes a sample-image generator.

Run:

```bash
python data/generate_samples.py
```

This generates overlapping sample images in:

```text
data/sample_images/
```

### 2. Run the panorama stitcher

Example using the provided sample images:

```bash
python main.py --images data/sample_images/newspaper1.jpg data/sample_images/newspaper2.jpg data/sample_images/newspaper3.jpg data/sample_images/newspaper4.jpg --output output/newspaper_panorama.jpg
```

The generated panorama will be saved to:

```text
output/newspaper_panorama.jpg
```

### 3. Use your own photos

You can provide your own overlapping images in left-to-right order:

```bash
python main.py --images photo1.jpg photo2.jpg photo3.jpg --output output/my_panorama.jpg
```

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--images` | Required | Two or more image paths in sequence |
| `--output` | `output/panorama.jpg` | Location where the panorama is saved |
| `--max-dimension` | `1000` | Maximum image dimension used during processing |
| `--ratio-thresh` | `0.75` | Lowe's ratio test threshold for feature matching |
| `--ransac-thresh` | `4.0` | RANSAC reprojection error threshold in pixels |

Run:

```bash
python main.py --help
```

to view the available command-line options.

## Testing

Unit tests cover:

- Image preprocessing
- Feature detection and matching
- Homography estimation
- Image stitching and blending
- Expected failure cases such as invalid inputs and insufficient matches

Run the complete test suite using:

```bash
python -m pytest tests/ -v
```

## Non-Functional Requirements Addressed

### Performance

Configurable image downscaling limits the computational cost of feature
detection and feature matching when processing large images.

### Reliability

The application validates inputs and reports processing failures such as
missing files, insufficient feature matches, and invalid homography
conditions.

### Usability

The system provides a simple command-line interface with clear arguments
and a `--help` option.

### Maintainability

The implementation follows a modular architecture where preprocessing,
feature detection, homography estimation, stitching, pipeline orchestration,
and logging are separated into individual modules.

### Error Handling

The system provides clear error messages for common failure conditions
instead of silently producing invalid results.

### Logging / Monitoring

Pipeline execution information is recorded both in the console and in:

```text
output/pipeline.log
```

## Screenshots / Results

A sample panorama generated by the system is available at:

```text
output/newspaper_panorama.jpg
```

The output demonstrates the result of feature detection, feature matching,
RANSAC-based homography estimation, perspective warping, and image blending.

## Design Approach

The project follows a feature-based image stitching approach.

First, input images are loaded and preprocessed. SIFT is then used to detect
distinctive local features and generate feature descriptors. Corresponding
features between adjacent images are identified using feature matching.

The matched points are used to estimate a homography using RANSAC. RANSAC
helps reduce the influence of incorrect feature correspondences.

The estimated transformation is then used to warp the images into a common
coordinate system. Finally, the aligned images are combined using feathered
blending to produce the panorama.

## Limitations

The quality of the panorama depends on factors such as:

- Amount of overlap between consecutive images
- Quality and distinctiveness of image features
- Image resolution
- Significant viewpoint changes
- Moving objects between images
- Large illumination differences

Images with insufficient overlap or too few reliable feature matches may
not produce a valid panorama.

## Future Enhancements

Possible future improvements include:

- Automatic exposure compensation
- Improved seam optimization
- Advanced blending techniques
- Support for larger image sequences
- Automatic cropping of empty panorama regions
- Additional feature detectors and descriptors
- Improved handling of moving objects
- Performance optimization for high-resolution images
- Graphical user interface
- Evaluation on larger real-world datasets

## Academic Project

This project was developed as part of the **CSE3010 – Computer Vision**
course project.

It demonstrates the application of computer vision concepts including
feature extraction, feature matching, projective geometry, homography,
RANSAC, image warping, and image blending.

## License

This project is developed for academic purposes as part of the Computer
Vision course project.