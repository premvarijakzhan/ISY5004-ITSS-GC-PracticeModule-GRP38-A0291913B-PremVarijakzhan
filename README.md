YOLOv12n Video Inference Tool
A comprehensive script for running object detection on videos using YOLOv12n models with performance comparison capabilities.
Overview
This tool allows you to:

Run object detection on videos using YOLOv12n variants
Compare performance between base and enhanced models
Generate visualization and performance metrics
Easily process videos with user-friendly interface options

Features

Multiple Model Support:

Original YOLOv12n base model
Enhanced YOLOv12n (with improvements like CBAM, etc.)


Performance Analysis:

FPS measurement and comparison
Processing time distribution
Relative speedup visualization between models
Detailed performance reports


Detection Options:

Adjustable confidence threshold
Configurable IoU threshold for NMS
Customizable image size


Video Processing Options:

Process entire videos or specific frame ranges
Real-time display option
Save processed videos with detections
Export detection results as JSON


Multiple Operation Modes:

Single model inference
Multi-model comparison
Interactive mode with guided options
Usage guide mode



Installation
Prerequisites

Python 3.8+
GPU support is optional but recommended for faster processing

Setup Environment
bash# Create a virtual environment
python -m venv yolo_env

# Activate it
# On Windows:
yolo_env\Scripts\activate
# On macOS/Linux:
source yolo_env/bin/activate

# Install required packages
pip install ultralytics opencv-python matplotlib tqdm
Directory Structure
Create the following directory structure:
yolo_project/
├── inference.py      # Main script
├── models/           # Store models here
│   ├── yolo12n.pt    # Base YOLOv12n model
│   └── enhanced_yolov12n.pt  # Your enhanced model
├── videos/           # Input videos
│   └── test_video.mp4
└── results/          # Output directory
Download Base Model
bash# For macOS/Linux:
wget -P models/ https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov12n.pt

# For Windows:
curl -L -o models/yolo12n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov12n.pt
Get Test Video (if needed)
bash# For macOS/Linux:
wget -P videos/ https://github.com/ultralytics/assets/raw/main/test_video.mp4

# For Windows:
curl -L -o videos/test_video.mp4 https://github.com/ultralytics/assets/raw/main/test_video.mp4
Usage
Basic Usage
Run inference with the base model:
bashpython inference.py --mode single --model base --video videos/test_video.mp4 --output results/output.mp4
Advanced Options
Process with custom detection settings:
bashpython inference.py --mode single --model base --video videos/test_video.mp4 --output results/output.mp4 --conf 0.4 --iou 0.5 --img-size 640
Compare base and enhanced models:
bashpython inference.py --mode compare --video videos/test_video.mp4 --output-dir results --max-frames 200
Interactive mode:
bashpython inference.py --mode interactive --video videos/test_video.mp4
Show usage guide:
bashpython inference.py --mode guide
Command-Line Arguments
ArgumentDescriptionDefault--modeOperation mode (single, compare, interactive, guide)single--modelModel to use in single mode (base, enhanced)base--base-modelPath to base YOLOv12n modelmodels/yolo12n.pt--enhanced-modelPath to enhanced modelmodels/enhanced_yolov12n.pt--videoPath to input videoRequired--outputPath to output video (single mode)None--output-dirDirectory to save comparison resultsresults--max-framesMaximum number of frames to processAll frames--start-frameFrame to start processing from0--confConfidence threshold (0-1)0.25--iouIoU threshold (0-1)0.45--img-sizeInput image size for model640--display/--no-displayWhether to display frames during processingTrue--show-plots/--no-plotsWhether to display comparison plotsTrue--save-detectionsSave detection results as JSONFalse--no-save-videosDo not save processed videos in compare modeFalse--verboseShow detailed progress informationTrue
Technical Details
Video Processing Pipeline

Initialization: Load selected models and configure detection settings
Frame Processing:

Read frame from video at original resolution
Process with YOLO model (handles resizing internally)
Draw bounding boxes and labels on original resolution


Output:

Save processed video with detections
Generate performance metrics and visualizations
Create comparison reports (when in compare mode)



Model Formats
The script handles two types of models:

Base Model: Standard PyTorch model using Ultralytics YOLO API
Enhanced Model: Your custom improved model also using Ultralytics YOLO API

Performance Metrics
The tool measures:

FPS (Frames Per Second): Higher is better, indicates inference speed
Processing Time: Time in milliseconds to process each frame
Processing Time Distribution: Min/max/average/variance of processing times
Relative Speedup: Performance gain relative to base model

Troubleshooting
Common Issues

Model Loading Fails:

Verify model paths are correct
Ensure models are downloaded/available
Check format compatibility


Display Issues:

Use --no-display if running in a headless environment
Check your OpenCV installation (pip install opencv-python instead of headless)


Slow Performance:

Try reducing --img-size for faster inference
Process fewer frames with --max-frames


Matplotlib Plot Display:

If plots don't display, try adding matplotlib.use('TkAgg') or install required backend
Use --no-plots to skip plot display but still save them as files



Examples
Basic Object Detection
bashpython inference.py --mode single --model base --video videos/street.mp4 --output results/detected.mp4
Processing a Specific Segment
bashpython inference.py --mode single --model base --video videos/long_video.mp4 --start-frame 1000 --max-frames 500 --output results/segment.mp4
Full Model Comparison
bashpython inference.py --mode compare --video videos/test.mp4 --output-dir results/comparison --max-frames 300 --show-plots
Detection with Custom Thresholds
bashpython inference.py --mode single --model base --video videos/test.mp4 --output results/high_conf.mp4 --conf 0.6 --iou 0.4
License
This project is provided as open-source software. Feel free to modify and distribute as needed.
Acknowledgments

Built with Ultralytics YOLO
Uses OpenCV for video processing and visualization
Visualization with Matplotlib


### Full Model Comparison

```bash
python inference.py --mode compare --video videos/test.mp4 --output-dir results/comparison --max-frames 300 --show-plots
Detection with Custom Thresholds
bashpython inference.py --mode single --model base --video videos/test.mp4 --output results/high_conf.mp4 --conf 0.6 --iou 0.4
License
This project is provided as open-source software. Feel free to modify and distribute as needed.
Acknowledgments

Built with Ultralytics YOLO
Uses OpenCV for video processing and visualization
Visualization with Matplotlib1000 --max-frames 500 --output results/segment.mp4


### Full Model Comparison

```bash
python inference.py --mode compare --video videos/test.mp4 --output-dir results/comparison --max-frames 300 --show-plots
Detection with Custom Thresholds
bashpython inference.py --mode single --model base --video videos/test.mp4 --output results/high_conf.mp4 --conf 0.6 --iou 0.4
License
This project is a part of NUS-ISS course. 
Acknowledgments

Built with Ultralytics YOLO
Uses OpenCV for video processing and visualization
Visualization with Matplotlib
