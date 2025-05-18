# YOLOv12n Video Inference Tool

A comprehensive tool for running object detection on videos using YOLOv12n models with performance comparison capabilities.

## Overview

This tool allows you to:

- Run object detection on videos using YOLOv12n variants
- Compare performance between base and enhanced models
- Generate visualization and performance metrics
- Easily process videos with user-friendly interface options

## Features

### Multiple Model Support
- Original YOLOv12n base model
- Enhanced YOLOv12n (with improvements like CBAM, etc.)

### Performance Analysis
- FPS measurement and comparison
- Processing time distribution
- Relative speedup visualization between models
- Detailed performance reports

### Detection Options
- Adjustable confidence threshold
- Configurable IoU threshold for NMS
- Customizable image size

### Video Processing Options
- Process entire videos or specific frame ranges
- Real-time display option
- Save processed videos with detections
- Export detection results as JSON

### Multiple Operation Modes
- Single model inference
- Multi-model comparison
- Interactive mode with guided options
- Usage guide mode

## Installation

### Prerequisites
- Python 3.8+
- GPU support is optional but recommended for faster processing

### Setup Environment
```bash
# Create a virtual environment
python -m venv yolo_env

# Activate it
# On Windows:
yolo_env\Scripts\activate
# On macOS/Linux:
source yolo_env/bin/activate

# Install required packages
pip install ultralytics opencv-python matplotlib tqdm
```

### Directory Structure
Create the following directory structure:
```
yolo_project/
├── inference.py      # Main script
├── models/           # Store models here
│   ├── yolo12n.pt    # Base YOLOv12n model
│   └── enhanced_yolov12n.pt  # Your enhanced model
├── videos/           # Input videos
│   └── test_video.mp4
└── results/          # Output directory
```

### Download Base Model
```bash
# For macOS/Linux:
wget -P models/ https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov12n.pt

# For Windows:
curl -L -o models/yolo12n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov12n.pt
```

### Get Test Video (if needed)
```bash
# For macOS/Linux:
wget -P videos/ https://github.com/ultralytics/assets/raw/main/test_video.mp4

# For Windows:
curl -L -o videos/test_video.mp4 https://github.com/ultralytics/assets/raw/main/test_video.mp4
```

## Usage

### Basic Usage
Run inference with the base model:
```bash
python inference.py --mode single --model base --video videos/test_video.mp4 --output results/output.mp4
```

### Advanced Options
Process with custom detection settings:
```bash
python inference.py --mode single --model base --video videos/test_video.mp4 --output results/output.mp4 --conf 0.4 --iou 0.5 --img-size 640
```

Compare base and enhanced models:
```bash
python inference.py --mode compare --video videos/test_video.mp4 --output-dir results --max-frames 200
```

Interactive mode:
```bash
python inference.py --mode interactive --video videos/test_video.mp4
```

Show usage guide:
```bash
python inference.py --mode guide
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Operation mode (single, compare, interactive, guide) | single |
| `--model` | Model to use in single mode (base, enhanced) | base |
| `--base-model` | Path to base YOLOv12n model | models/yolo12n.pt |
| `--enhanced-model` | Path to enhanced model | models/enhanced_yolov12n.pt |
| `--video` | Path to input video | Required |
| `--output` | Path to output video (single mode) | None |
| `--output-dir` | Directory to save comparison results | results |
| `--max-frames` | Maximum number of frames to process | All frames |
| `--start-frame` | Frame to start processing from | 0 |
| `--conf` | Confidence threshold (0-1) | 0.25 |
| `--iou` | IoU threshold (0-1) | 0.45 |
| `--img-size` | Input image size for model | 640 |
| `--display/--no-display` | Whether to display frames during processing | True |
| `--show-plots/--no-plots` | Whether to display comparison plots | True |
| `--save-detections` | Save detection results as JSON | False |
| `--no-save-videos` | Do not save processed videos in compare mode | False |
| `--verbose` | Show detailed progress information | True |

## Technical Details

### Video Processing Pipeline
1. **Initialization**: Load selected models and configure detection settings
2. **Frame Processing**:
   - Read frame from video at original resolution
   - Process with YOLO model (handles resizing internally)
   - Draw bounding boxes and labels on original resolution
3. **Output**:
   - Save processed video with detections
   - Generate performance metrics and visualizations
   - Create comparison reports (when in compare mode)

### Model Formats
The script handles two types of models:
- **Base Model**: Standard PyTorch model using Ultralytics YOLO API
- **Enhanced Model**: Your custom improved model also using Ultralytics YOLO API

### Performance Metrics
The tool measures:
- **FPS (Frames Per Second)**: Higher is better, indicates inference speed
- **Processing Time**: Time in milliseconds to process each frame
- **Processing Time Distribution**: Min/max/average/variance of processing times
- **Relative Speedup**: Performance gain relative to base model

## Troubleshooting

### Common Issues
- **Model Loading Fails**:
  - Verify model paths are correct
  - Ensure models are downloaded/available
  - Check format compatibility

- **Display Issues**:
  - Use `--no-display` if running in a headless environment
  - Check your OpenCV installation (pip install opencv-python instead of headless)

- **Slow Performance**:
  - Try reducing `--img-size` for faster inference
  - Process fewer frames with `--max-frames`

- **Matplotlib Plot Display**:
  - If plots don't display, try adding `matplotlib.use('TkAgg')` or install required backend
  - Use `--no-plots` to skip plot display but still save them as files

## Quick Reference Commands

### Setup Commands
```bash
# Create and activate virtual environment (Windows)
python -m venv yolo_env
yolo_env\Scripts\activate

# Create and activate virtual environment (macOS/Linux)
python -m venv yolo_env
source yolo_env/bin/activate

# Install dependencies
pip install ultralytics opencv-python matplotlib tqdm

# Download base YOLOv12n model (Windows)
curl -L -o models/yolo12n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov12n.pt

# Download test video (Windows)
curl -L -o videos/test_video.mp4 https://github.com/ultralytics/assets/raw/main/test_video.mp4
```

### Ready-to-Use Inference Commands
```bash
# Basic inference with base model
python inference.py --mode single --model base --video videos/test_video.mp4 --output results/output.mp4

# Basic inference with enhanced model
python inference.py --mode single --model enhanced --video videos/test_video.mp4 --output results/enhanced_output.mp4

# Side-by-side model comparison (300 frames)
python inference.py --mode compare --video videos/test_video.mp4 --output-dir results/comparison --max-frames 300 --show-plots

# High-precision detection (higher confidence, lower IoU)
python inference.py --mode single --model base --video videos/test_video.mp4 --output results/high_precision.mp4 --conf 0.6 --iou 0.3

# High-recall detection (lower confidence, higher IoU)
python inference.py --mode single --model base --video videos/test_video.mp4 --output results/high_recall.mp4 --conf 0.2 --iou 0.6

# Fast processing with smaller image size
python inference.py --mode single --model base --video videos/test_video.mp4 --output results/faster.mp4 --img-size 416

# Process video segment (frames 1000-1500)
python inference.py --mode single --model base --video videos/test_video.mp4 --start-frame 1000 --max-frames 500 --output results/segment.mp4

# Interactive mode for guided operation
python inference.py --mode interactive --video videos/test_video.mp4

# Save detection results as JSON
python inference.py --mode single --model base --video videos/test_video.mp4 --output results/output.mp4 --save-detections
```

## Usage Examples

### Basic Object Detection
```bash
python inference.py --mode single --model base --video videos/street.mp4 --output results/detected.mp4
```

### Processing a Specific Segment
```bash
python inference.py --mode single --model base --video videos/long_video.mp4 --start-frame 1000 --max-frames 500 --output results/segment.mp4
```

### Full Model Comparison
```bash
python inference.py --mode compare --video videos/test.mp4 --output-dir results/comparison --max-frames 300 --show-plots
```

### Detection with Custom Thresholds
```bash
python inference.py --mode single --model base --video videos/test.mp4 --output results/high_conf.mp4 --conf 0.6 --iou 0.4
```

## License
This project is a part of NUS-ISS course.

## Acknowledgments
- Built with Ultralytics YOLO
- Uses OpenCV for video processing and visualization
- Visualization with Matplotlib