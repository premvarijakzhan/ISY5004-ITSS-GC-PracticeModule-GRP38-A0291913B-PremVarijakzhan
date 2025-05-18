import os
import time
import torch
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import argparse
import sys
import json


class ModelInference:
    def __init__(self, conf_threshold=0.25, iou_threshold=0.45, img_size=640):
        # Available models to select from
        self.models = {
            'base': None,      # Original YOLOv12n
            'enhanced': None,  # Enhanced YOLOv12n
            'quantized': None  # Quantized INT8 model
        }

        self.model_paths = {
            'base': 'models/base_yolo12n.pt',  # Default YOLO model
            'enhanced': None,             # Will be set when loaded
            'quantized': None             # Will be set when loaded
        }

        self.current_model_name = None
        self.current_model = None
        self.class_names = None
        self.device = 'cpu'  # Force CPU for edge device simulation

        # Inference settings
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size

        # Performance tracking
        self.fps_stats = {}

        # Class colors for visualization
        self.colors = {}

    def set_thresholds(self, conf_threshold=None, iou_threshold=None, img_size=None):
        """Update detection thresholds"""
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
        if img_size is not None:
            self.img_size = img_size

        print(
            f"Updated settings: conf={self.conf_threshold}, iou={self.iou_threshold}, img_size={self.img_size}")
        return True

    def load_model(self, model_type, model_path=None):
        """
        Load a specific model type

        Args:
            model_type: 'base', 'enhanced', or 'quantized'
            model_path: Path to model weights (if None, use default)
        """
        if model_type not in self.models:
            print(f"Invalid model type: {model_type}")
            return False

        if model_path:
            self.model_paths[model_type] = model_path

        try:
            # Always force CPU loading for edge device simulation
            print(
                f"Loading {model_type} model from {self.model_paths[model_type]}")

            if model_type == 'base' or model_type == 'enhanced':
                # Load standard YOLO model
                self.models[model_type] = YOLO(self.model_paths[model_type])
                # Force CPU
                self.models[model_type].to('cpu')

            elif model_type == 'quantized':
                # For quantized model, we need special handling
                try:
                    # Try loading as ONNX
                    if self.model_paths[model_type].endswith('.onnx'):
                        import onnxruntime as ort
                        print("Loading ONNX model")
                        self.models[model_type] = ort.InferenceSession(
                            self.model_paths[model_type],
                            providers=['CPUExecutionProvider']
                        )
                    else:
                        # Try loading as PyTorch quantized model
                        print("Loading quantized PyTorch model")
                        self.models[model_type] = torch.jit.load(
                            self.model_paths[model_type])
                except Exception as e:
                    print(f"Failed to load quantized model: {e}")
                    print("Falling back to standard YOLO loading")
                    self.models[model_type] = YOLO(
                        self.model_paths[model_type])
                    self.models[model_type].to('cpu')

            # Set as current model
            self.current_model_name = model_type
            self.current_model = self.models[model_type]

            # Get class names
            if model_type in ['base', 'enhanced']:
                self.class_names = self.models[model_type].names

                # Generate colors for classes if not already done
                if not self.colors:
                    np.random.seed(42)  # For reproducibility
                    for class_id in self.class_names:
                        self.colors[class_id] = [
                            int(c) for c in np.random.randint(0, 255, size=3)]

            print(f"Successfully loaded {model_type} model")
            return True

        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
            return False

    def set_current_model(self, model_type):
        """Set which model to use for inference"""
        if model_type not in self.models or self.models[model_type] is None:
            print(f"Model {model_type} not loaded yet")
            return False

        self.current_model_name = model_type
        self.current_model = self.models[model_type]
        print(f"Set current model to {model_type}")
        return True

    def process_image(self, image):
        """
        Process a single image with the current model

        Args:
            image: numpy array or PIL Image

        Returns:
            processed_image: image with detections
            detections: detection results
            fps: frames per second
        """
        if self.current_model is None:
            print("No model loaded")
            return image, None, 0

        start_time = time.time()
        original_size = image.shape[:2]  # (height, width)

        # For standard YOLO models
        if self.current_model_name in ['base', 'enhanced']:
            # YOLO model automatically handles resizing internally
            results = self.current_model(
                image, conf=self.conf_threshold, iou=self.iou_threshold)
            processed_image = results[0].plot()
            detections = results[0]

        # For quantized ONNX models
        elif self.current_model_name == 'quantized' and hasattr(self.current_model, 'run'):
            # Convert image to ONNX input format (resizes to self.img_size)
            input_img = self._preprocess_for_onnx(image)

            # Run inference
            # outputs = self.current_model.run(None, {'images': input_img})
            outputs = self.current_model.run(None, {'input': input_img})

            # Process outputs (draws on original size image)
            processed_image = self._draw_onnx_detections(image, outputs)
            detections = outputs

        # For quantized PyTorch models
        else:
            # Assume TorchScript model
            input_tensor = self._preprocess_for_torch(image)

            with torch.no_grad():
                outputs = self.current_model(input_tensor)

            # Process outputs (draws on original size image)
            processed_image = self._draw_torch_detections(image, outputs)
            detections = outputs

        inference_time = time.time() - start_time
        fps = 1.0 / inference_time

        # Update FPS stats
        if self.current_model_name not in self.fps_stats:
            self.fps_stats[self.current_model_name] = []
        self.fps_stats[self.current_model_name].append(fps)

        return processed_image, detections, fps

    def process_video(self, video_path, output_path=None, display=True, max_frames=None,
                      start_frame=0, verbose=True, save_detections=False, detection_path=None,
                      rotate=0):
        """
        Process a video with the current model

        Args:
            video_path: Path to input video
            output_path: Path to save output video (None for no saving)
            display: Whether to display frames during processing
            max_frames: Maximum number of frames to process (None for all)
            start_frame: Frame to start processing from
            verbose: Whether to show progress bar and output stats
            save_detections: Whether to save detection results
            detection_path: Path to save detection results (None for default)
            rotate: Rotation angle (0, 90, 180, 270) to apply to video

        Returns:
            Dictionary with performance statistics
        """
        if self.current_model is None:
            print("No model loaded")
            return

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            return

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames is not None:
            total_frames = min(total_frames, start_frame + max_frames)

        if verbose:
            print(
                f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")

        # Skip to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            if verbose:
                print(f"Starting from frame {start_frame}")

        # Initialize video writer if saving
        video_writer = None
        if output_path:
            os.makedirs(os.path.dirname(
                os.path.abspath(output_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_path, fourcc, fps, (width, height))

        # Initialize detection storage if saving
        all_detections = [] if save_detections else None

        # Process frames
        frame_count = 0
        processing_times = []

        if verbose:
            pbar = tqdm(total=total_frames - start_frame,
                        desc=f"Processing video with {self.current_model_name} model")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Apply rotation if specified
            if rotate == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotate == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotate == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Process frame
            processed_frame, detections, frame_fps = self.process_image(frame)
            processing_times.append(1000 / frame_fps)  # Convert to ms

            # Save detections if requested
            if save_detections and hasattr(detections, 'boxes'):
                frame_detections = {
                    'frame': start_frame + frame_count,
                    'boxes': detections.boxes.cpu().numpy().tolist() if hasattr(detections.boxes, 'cpu') else [],
                    'labels': detections.boxes.cls.cpu().numpy().tolist() if hasattr(detections.boxes, 'cls') else [],
                    'confidences': detections.boxes.conf.cpu().numpy().tolist() if hasattr(detections.boxes, 'conf') else []
                }
                all_detections.append(frame_detections)

            # Add FPS info
            cv2.putText(
                processed_frame,
                f"{self.current_model_name}: {frame_fps:.1f} FPS",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            # Save frame
            if video_writer:
                video_writer.write(processed_frame)

            # Display frame
            if display:
                cv2.imshow('Processed Frame', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1
            if verbose:
                pbar.update(1)

            if max_frames is not None and frame_count >= max_frames:
                break

        if verbose:
            pbar.close()
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

        # Save detections if requested
        if save_detections and all_detections:
            if detection_path is None:
                detection_path = os.path.splitext(
                    output_path)[0] + '_detections.json' if output_path else 'detections.json'

            with open(detection_path, 'w') as f:
                json.dump(all_detections, f)

            if verbose:
                print(f"Saved detections to {detection_path}")

        # Report performance
        avg_time = np.mean(processing_times)
        std_time = np.std(processing_times)
        avg_fps = 1000 / avg_time

        if verbose:
            print(f"\nProcessed {frame_count} frames")
            print(
                f"Average processing time: {avg_time:.2f} ms (±{std_time:.2f} ms)")
            print(f"Average FPS: {avg_fps:.2f}")

        # Return processing stats
        return {
            'frame_count': frame_count,
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'avg_fps': avg_fps,
            'times_ms': processing_times
        }

    def compare_models(self, video_path, output_dir, max_frames=100, show_plots=True,
                       start_frame=0, save_videos=True, save_detections=False,
                       conf_threshold=None, iou_threshold=None, rotate=0):
        """
        Compare performance of all loaded models on the same video

        Args:
            video_path: Path to input video
            output_dir: Directory to save results
            max_frames: Maximum frames to process per model
            show_plots: Whether to display plots
            start_frame: Frame to start processing from
            save_videos: Whether to save processed videos
            save_detections: Whether to save detection results
            conf_threshold: Override confidence threshold
            iou_threshold: Override IoU threshold
            rotate: Rotation angle (0, 90, 180, 270) to apply to video
        """
        os.makedirs(output_dir, exist_ok=True)

        # Temporarily update thresholds if provided
        original_conf = self.conf_threshold
        original_iou = self.iou_threshold

        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold

        # Results to collect
        results = {}

        # Process video with each loaded model
        for model_name, model in self.models.items():
            if model is None:
                print(f"Model {model_name} not loaded, skipping")
                continue

            print(f"\n--- Testing {model_name} model ---")

            # Set as current model
            self.set_current_model(model_name)

            # Process video
            output_path = os.path.join(
                output_dir, f"{model_name}_output.mp4") if save_videos else None
            detection_path = os.path.join(
                output_dir, f"{model_name}_detections.json") if save_detections else None

            stats = self.process_video(
                video_path=video_path,
                output_path=output_path,
                display=False,
                max_frames=max_frames,
                start_frame=start_frame,
                save_detections=save_detections,
                detection_path=detection_path,
                rotate=rotate
            )

            # Store results
            results[model_name] = stats

        # Create comparison report
        self._generate_comparison_report(results, output_dir, show_plots)

        # Restore original thresholds
        self.conf_threshold = original_conf
        self.iou_threshold = original_iou

        return results

    def _generate_comparison_report(self, results, output_dir, show_plots=True):
        """Generate performance comparison report"""
        # Create comparison plots
        plt.figure(figsize=(12, 6))

        # FPS comparison
        models = list(results.keys())
        avg_fps = [results[m]['avg_fps'] for m in models]

        plt.bar(models, avg_fps, color=['blue', 'green', 'red'][:len(models)])
        plt.ylabel('Frames Per Second')
        plt.title('Inference Speed Comparison')

        # Add numerical values on bars
        for i, v in enumerate(avg_fps):
            plt.text(i, v + 0.5, f"{v:.1f}", ha='center')

        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'fps_comparison.png'), dpi=300)
        if show_plots:
            plt.show()
        plt.close()

        # Processing time distribution
        plt.figure(figsize=(12, 6))

        # Create box plots of processing times
        times_data = [results[m]['times_ms'] for m in models]
        plt.boxplot(times_data, labels=models)

        plt.ylabel('Processing Time (ms)')
        plt.title('Processing Time Distribution')
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'time_distribution.png'), dpi=300)
        if show_plots:
            plt.show()
        plt.close()

        # Speed improvement bar chart (if multiple models)
        if len(models) > 1:
            plt.figure(figsize=(10, 6))

            # Calculate speedups compared to base or slowest model
            base_model = 'base' if 'base' in results else models[0]
            base_fps = results[base_model]['avg_fps']

            # Calculate speedup for each model
            speedups = [(m, results[m]['avg_fps'] / base_fps)
                        for m in models if m != base_model]

            if speedups:
                labels = [m for m, _ in speedups]
                values = [s for _, s in speedups]

                plt.bar(labels, values, color=['green', 'red'][:len(speedups)])
                plt.ylabel(f'Speedup compared to {base_model}')
                plt.title('Relative Speed Improvement')

                # Add numerical values on bars
                for i, v in enumerate(values):
                    plt.text(i, v + 0.05, f"{v:.2f}x", ha='center')

                plt.grid(axis='y', alpha=0.3)
                plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.3)
                plt.savefig(os.path.join(
                    output_dir, 'speedup_comparison.png'), dpi=300)
                if show_plots:
                    plt.show()
                plt.close()

        # Create text report
        report_path = os.path.join(output_dir, 'performance_report.txt')
        with open(report_path, 'w') as f:
            f.write("=== Model Performance Comparison ===\n\n")

            for model_name, stats in results.items():
                f.write(f"--- {model_name.upper()} MODEL ---\n")
                f.write(f"Average FPS: {stats['avg_fps']:.2f}\n")
                f.write(
                    f"Average processing time: {stats['avg_time_ms']:.2f} ms (±{stats['std_time_ms']:.2f} ms)\n")
                f.write(f"Processed frames: {stats['frame_count']}\n\n")

            # Speed comparison
            if 'base' in results and 'quantized' in results:
                speedup = results['quantized']['avg_fps'] / \
                    results['base']['avg_fps']
                f.write(f"Quantized model speedup over base: {speedup:.2f}x\n")

            if 'enhanced' in results and 'quantized' in results:
                speedup = results['quantized']['avg_fps'] / \
                    results['enhanced']['avg_fps']
                f.write(
                    f"Quantized model speedup over enhanced: {speedup:.2f}x\n")

            # Settings used
            f.write(f"\nDetection Settings:\n")
            f.write(f"Confidence threshold: {self.conf_threshold}\n")
            f.write(f"IoU threshold: {self.iou_threshold}\n")
            f.write(f"Image size: {self.img_size}\n")

            # Date and time
            from datetime import datetime
            f.write(
                f"\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"Performance report saved to {report_path}")

    # Helper methods for different model formats
    def _preprocess_for_onnx(self, image):
        """Preprocess image for ONNX model input"""
        # Convert BGR to RGB
        if isinstance(image, np.ndarray) and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        img = cv2.resize(image, (self.img_size, self.img_size))

        # Normalize and transpose
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC to CHW

        # Add batch dimension
        img = np.expand_dims(img, 0)

        return img

    def _preprocess_for_torch(self, image):
        """Preprocess image for PyTorch model input"""
        # Convert BGR to RGB
        if isinstance(image, np.ndarray) and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        img = cv2.resize(image, (self.img_size, self.img_size))

        # Convert to tensor
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        # Add batch dimension
        img = img.unsqueeze(0)

        return img

    def _draw_onnx_detections(self, image, outputs):
        """Draw detections from ONNX model outputs"""
        # Create a copy of the input image for drawing
        result_img = image.copy()
        h, w = result_img.shape[:2]

        # Standard YOLO ONNX model typically outputs:
        # - outputs[0]: detection boxes, scores, and class ids combined in shape [1, num_boxes, 7]
        #   where each detection is [batch_id, x1, y1, x2, y2, score, class_id]

        # Extract detections - adapt this based on your specific ONNX model format
        if len(outputs) >= 1 and isinstance(outputs[0], np.ndarray):
            # For YOLO ONNX standard output format
            detections = outputs[0]

            # Filter detections by confidence threshold
            valid_detections = detections[detections[:,
                                                     :, 4] > self.conf_threshold]
            if len(valid_detections.shape) == 3:
                # Remove batch dimension
                valid_detections = valid_detections[0]

            # Apply NMS per class using cv2's NMS implementation
            boxes = []
            class_ids = []
            confidences = []

            if len(valid_detections) > 0:
                # Group detections by class
                for class_id in np.unique(valid_detections[:, 5].astype(np.int32)):
                    class_detections = valid_detections[valid_detections[:, 5].astype(
                        np.int32) == class_id]

                    # Prepare data for NMS
                    class_boxes = class_detections[:, :4]  # x1, y1, x2, y2
                    class_confidences = class_detections[:, 4]

                    # Apply NMS
                    indices = cv2.dnn.NMSBoxes(
                        class_boxes.tolist(),
                        class_confidences.tolist(),
                        self.conf_threshold,
                        self.iou_threshold
                    )

                    # Convert to 1D array if needed (for OpenCV 4.5.4+)
                    if len(indices) > 0 and isinstance(indices, np.ndarray) and indices.ndim > 1:
                        indices = indices.flatten()

                    # Add to final detections
                    for idx in indices:
                        boxes.append(class_boxes[idx])
                        confidences.append(class_confidences[idx])
                        class_ids.append(class_id)

                # Draw final detections
                for box, class_id, confidence in zip(boxes, class_ids, confidences):
                    # Scale coordinates to original image
                    x1, y1, x2, y2 = box

                    # Check if the box coordinates are normalized
                    if max(x1, y1, x2, y2) <= 1.0:
                        x1, y1, x2, y2 = int(
                            x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                    else:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Get class name and color
                    class_name = str(class_id)
                    if hasattr(self, 'class_names') and self.class_names and class_id in self.class_names:
                        class_name = self.class_names[class_id]

                    color = self.colors.get(class_id, [0, 255, 0])

                    # Draw box and label
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(result_img, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return result_img

    def _draw_torch_detections(self, image, outputs):
        """Draw detections from PyTorch model outputs"""
        # Create a copy of the input image for drawing
        result_img = image.copy()
        h, w = result_img.shape[:2]

        # Convert outputs to numpy for easier processing
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu().numpy()

        # Typical YOLO output format: [batch, num_detections, 6]
        # where each detection is [x1, y1, x2, y2, confidence, class_id]
        if len(outputs.shape) == 3 and outputs.shape[2] >= 6:
            detections = outputs[0]  # Remove batch dimension

            # Filter by confidence threshold
            mask = detections[:, 4] > self.conf_threshold
            filtered_detections = detections[mask]

            # Apply NMS per class using torchvision NMS or cv2's NMS
            boxes = filtered_detections[:, :4]
            confidences = filtered_detections[:, 4]
            class_ids = filtered_detections[:, 5].astype(int)

            # Group by class for NMS
            final_boxes = []
            final_confidences = []
            final_class_ids = []

            for cls in np.unique(class_ids):
                cls_mask = class_ids == cls
                cls_boxes = boxes[cls_mask]
                cls_confidences = confidences[cls_mask]

                # Apply NMS using cv2.dnn
                indices = cv2.dnn.NMSBoxes(
                    cls_boxes.tolist(),
                    cls_confidences.tolist(),
                    self.conf_threshold,
                    self.iou_threshold
                )

                # Handle different output formats of NMSBoxes
                if len(indices) > 0:
                    if isinstance(indices, np.ndarray) and indices.ndim > 1:
                        indices = indices.flatten()

                    for idx in indices:
                        final_boxes.append(cls_boxes[idx])
                        final_confidences.append(cls_confidences[idx])
                        final_class_ids.append(cls)

            # Draw final detections
            for box, class_id, confidence in zip(final_boxes, final_class_ids, final_confidences):
                x1, y1, x2, y2 = box

                # Check if coordinates are normalized
                if x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0:  # Normalized
                    x1, y1, x2, y2 = int(
                        x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                else:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get class name and color
                class_name = str(class_id)
                if hasattr(self, 'class_names') and self.class_names and class_id in self.class_names:
                    class_name = self.class_names[class_id]

                color = self.colors.get(class_id, [0, 255, 0])

                # Draw box and label
                cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(result_img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return result_img

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for two boxes"""
        # Assuming box format is [x1, y1, x2, y2]
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Calculate intersection area
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)

        if x_max < x_min or y_max < y_min:
            return 0.0

        intersection = (x_max - x_min) * (y_max - y_min)

        # Calculate union area
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='YOLO Model Inference and Comparison',
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Mode of operation
    parser.add_argument('--mode', type=str, choices=['single', 'compare', 'interactive', 'guide'], default='single',
                        help='''Mode of operation:
single: Process video with a single model
compare: Compare performance of multiple models
interactive: Choose options through an interactive menu
guide: Show usage guide and exit''')

    # Model settings
    parser.add_argument('--model', type=str, choices=['base', 'enhanced', 'quantized'], default='base',
                        help='Model to use for inference in single mode')
    parser.add_argument('--base-model', type=str, default='models/base_yolo12n.pt',
                        help='Path to base Fine-tuned YOLOv12n model')
    parser.add_argument('--enhanced-model', type=str,
                        default='models/enhanced_yolov12n.pt',
                        help='Path to enhanced YOLOv12n model')
    parser.add_argument('--quantized-model', type=str,
                        default='models/enhanced_yolov12n_quantized_int8.onnx',
                        help='Path to quantized model')

    # Video settings
    parser.add_argument('--video', type=str, default=None,
                        help='Path to input video')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output video (for single mode)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save comparison results')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to process')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Frame to start processing from')

    # Detection settings
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (0-1)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold (0-1)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size for model')

    # Display settings
    parser.add_argument('--display', action='store_true',
                        help='Display processed frames')
    parser.add_argument('--no-display', dest='display', action='store_false',
                        help='Do not display processed frames')
    parser.add_argument('--show-plots', action='store_true',
                        help='Show comparison plots')
    parser.add_argument('--no-plots', dest='show_plots', action='store_false',
                        help='Do not show comparison plots')

    # Video orientation
    parser.add_argument('--rotate', type=int, choices=[0, 90, 180, 270], default=180,
                        help='Rotate video by specified degrees')

    # Advanced options
    parser.add_argument('--save-detections', action='store_true',
                        help='Save detection results as JSON')
    parser.add_argument('--detection-path', type=str, default=None,
                        help='Path to save detection results (for single mode)')
    parser.add_argument('--no-save-videos', dest='save_videos', action='store_false',
                        help='Do not save processed videos in compare mode')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed progress and information')

    # Set defaults
    parser.set_defaults(display=True, show_plots=True,
                        save_videos=True, verbose=True)

    # Process arguments
    args = parser.parse_args()

    # Handle guide mode first
    if args.mode == 'guide':
        show_guide()
        sys.exit(0)

    # Validate arguments
    if args.mode != 'interactive' and args.video is None:
        parser.error("--video is required for non-interactive modes")

    return args


def show_guide():
    """Display a comprehensive usage guide"""
    guide_text = """
=== YOLOv12n Video Inference Tool - Usage Guide ===

This tool allows you to run object detection on videos using different YOLO models:
- Original YOLOv12n (base)
- Enhanced YOLOv12n with CBAM and transformers (enhanced)
- INT8 quantized YOLOv12n for edge devices (quantized)

=== Modes of Operation ===

1. Single Model Mode:
   Process a video with one specific model and save/display the results.
   
   Example:
   python inference_with_quant.py --mode single --model base --video test.mp4 --output results/base_output.mp4

2. Comparison Mode:
   Compare the performance of multiple models on the same video.
   
   Example:
   python inference_with_quant.py --mode compare --video test.mp4 --output-dir results

3. Interactive Mode:
   Run the script with an interactive menu to choose options on the fly.
   
   Example:
   python inference_with_quant.py --mode interactive --video test.mp4

=== Model Paths ===

You need to specify paths to your model files:

--base-model: Path to original Fine-tuned YOLOv12n model (default: models/base_yolo12n.pt)
--enhanced-model: Path to your enhanced model
--quantized-model: Path to your quantized model

=== Detection Settings ===

--conf: Confidence threshold (0-1), default: 0.25
   - Higher values show fewer but more confident detections
   - Lower values show more detections including less confident ones

--iou: Intersection over Union threshold (0-1), default: 0.45
   - Controls non-maximum suppression strictness
   - Higher values allow more overlapping boxes

--img-size: Input image size for the model, default: 640
   - Larger sizes give better accuracy but slower inference
   - Smaller sizes give faster inference but may reduce accuracy

=== Video Processing Options ===

--max-frames: Limit the number of frames to process
--start-frame: Start processing from this frame number
--display/--no-display: Show/hide processed frames in a window
--save-detections: Save detection results as JSON for further analysis
--rotate: Rotate video by specified degrees (0, 90, 180, 270)

=== Performance Evaluation ===

Performance metrics calculated include:
- Processing speed (FPS)
- Processing time per frame
- Relative speedup between models

The comparison report includes:
- Graphs of FPS and processing time
- Text report with detailed statistics
- Visualizations of the detection results

=== Examples ===

1. Basic comparison with default settings:
   python inference_with_quant.py --mode compare --video videos/test_video.mp4

2. Detailed single model processing:
   python inference_with_quant.py --mode single --model quantized --video videos/test_video.mp4 --output results/results.mp4 --conf 0.3 --iou 0.5 --save-detections

3. Process specific frames with no display:
   python inference_with_quant.py --mode single --model enhanced --video videos/long_video.mp4 --start-frame 1000 --max-frames 500 --no-display

4. Compare models with custom thresholds and no plots:
   python inference_with_quant.py --mode compare --video videos/test_video.mp4 --conf 0.4 --iou 0.6 --no-plots

5. Fix upside-down video:
   python inference_with_quant.py --mode single --model base --video videos/test_video.mp4 --rotate 180 --output results/fixed_output.mp4

=== Notes ===

- All models run on CPU to simulate edge device performance
- Quantized models will generally provide faster inference with slightly lower accuracy
- The enhanced model may provide better detection quality but could be slower than the base model
"""
    print(guide_text)


def interactive_mode(inference_engine, video_path, output_dir):
    """Run interactive mode where user can choose models and options"""
    os.makedirs(output_dir, exist_ok=True)

    if video_path is None:
        video_path = input("Enter path to video file: ")

    while True:
        print("\n=== YOLO Model Inference ===")
        print("1. Run inference with base model")
        print("2. Run inference with enhanced model")
        print("3. Run inference with quantized model")
        print("4. Compare all available models")
        print("5. Change detection settings")
        print("6. Change video")
        print("7. Show loaded models")
        print("8. Show guide")
        print("9. Exit")

        choice = input("\nEnter your choice (1-9): ")

        if choice == '1':
            if inference_engine.models['base'] is None:
                print("Base model not loaded. Please specify path.")
                model_path = input("Enter path to base model: ")
                inference_engine.load_model('base', model_path)

            inference_engine.set_current_model('base')
            output_path = os.path.join(output_dir, 'base_output.mp4')
            display = input("Display frames? (y/n): ").lower() == 'y'
            max_frames = input("Maximum frames (0 for all): ")
            max_frames = int(max_frames) if max_frames.isdigit() and int(
                max_frames) > 0 else None
            start_frame = input("Start from frame (0 for beginning): ")
            start_frame = int(start_frame) if start_frame.isdigit() else 0
            save_detections = input("Save detections? (y/n): ").lower() == 'y'
            rotate = input("Rotate video? (0, 90, 180, 270): ")
            rotate = int(rotate) if rotate in ['0', '90', '180', '270'] else 0

            inference_engine.process_video(
                video_path=video_path,
                output_path=output_path,
                display=display,
                max_frames=max_frames,
                start_frame=start_frame,
                save_detections=save_detections,
                rotate=rotate
            )

        elif choice == '2':
            if inference_engine.models['enhanced'] is None:
                print("Enhanced model not loaded. Please specify path.")
                model_path = input("Enter path to enhanced model: ")
                inference_engine.load_model('enhanced', model_path)

            inference_engine.set_current_model('enhanced')
            output_path = os.path.join(output_dir, 'enhanced_output.mp4')
            display = input("Display frames? (y/n): ").lower() == 'y'
            max_frames = input("Maximum frames (0 for all): ")
            max_frames = int(max_frames) if max_frames.isdigit() and int(
                max_frames) > 0 else None
            start_frame = input("Start from frame (0 for beginning): ")
            start_frame = int(start_frame) if start_frame.isdigit() else 0
            save_detections = input("Save detections? (y/n): ").lower() == 'y'
            rotate = input("Rotate video? (0, 90, 180, 270): ")
            rotate = int(rotate) if rotate in ['0', '90', '180', '270'] else 0

            inference_engine.process_video(
                video_path=video_path,
                output_path=output_path,
                display=display,
                max_frames=max_frames,
                start_frame=start_frame,
                save_detections=save_detections,
                rotate=rotate
            )

        elif choice == '3':
            if inference_engine.models['quantized'] is None:
                print("Quantized model not loaded. Please specify path.")
                model_path = input("Enter path to quantized model: ")
                inference_engine.load_model('quantized', model_path)

            inference_engine.set_current_model('quantized')
            output_path = os.path.join(output_dir, 'quantized_output.mp4')
            display = input("Display frames? (y/n): ").lower() == 'y'
            max_frames = input("Maximum frames (0 for all): ")
            max_frames = int(max_frames) if max_frames.isdigit() and int(
                max_frames) > 0 else None
            start_frame = input("Start from frame (0 for beginning): ")
            start_frame = int(start_frame) if start_frame.isdigit() else 0
            save_detections = input("Save detections? (y/n): ").lower() == 'y'
            rotate = input("Rotate video? (0, 90, 180, 270): ")
            rotate = int(rotate) if rotate in ['0', '90', '180', '270'] else 0

            inference_engine.process_video(
                video_path=video_path,
                output_path=output_path,
                display=display,
                max_frames=max_frames,
                start_frame=start_frame,
                save_detections=save_detections,
                rotate=rotate
            )

        elif choice == '4':
            # Check if at least two models are loaded
            loaded_models = sum(
                1 for m in inference_engine.models.values() if m is not None)
            if loaded_models < 2:
                print("Need at least two models loaded for comparison")
                continue

            max_frames = input("Maximum frames per model (0 for all): ")
            max_frames = int(max_frames) if max_frames.isdigit() and int(
                max_frames) > 0 else None
            start_frame = input("Start from frame (0 for beginning): ")
            start_frame = int(start_frame) if start_frame.isdigit() else 0
            show_plots = input("Show plots? (y/n): ").lower() == 'y'
            save_videos = input(
                "Save processed videos? (y/n): ").lower() == 'y'
            save_detections = input("Save detections? (y/n): ").lower() == 'y'

            conf_override = input(
                "Override confidence threshold? (enter value or leave empty): ")
            conf_threshold = float(conf_override) if conf_override and 0 <= float(
                conf_override) <= 1 else None

            iou_override = input(
                "Override IoU threshold? (enter value or leave empty): ")
            iou_threshold = float(iou_override) if iou_override and 0 <= float(
                iou_override) <= 1 else None

            rotate = input("Rotate video? (0, 90, 180, 270): ")
            rotate = int(rotate) if rotate in ['0', '90', '180', '270'] else 0

            inference_engine.compare_models(
                video_path=video_path,
                output_dir=output_dir,
                max_frames=max_frames,
                show_plots=show_plots,
                start_frame=start_frame,
                save_videos=save_videos,
                save_detections=save_detections,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                rotate=rotate
            )

        elif choice == '5':
            print(
                f"Current settings: conf={inference_engine.conf_threshold}, iou={inference_engine.iou_threshold}, img_size={inference_engine.img_size}")
            conf = input(
                "Enter new confidence threshold (0-1, or press Enter to keep current): ")
            iou = input(
                "Enter new IoU threshold (0-1, or press Enter to keep current): ")
            img_size = input(
                "Enter new image size (or press Enter to keep current): ")

            conf_val = float(conf) if conf and 0 <= float(conf) <= 1 else None
            iou_val = float(iou) if iou and 0 <= float(iou) <= 1 else None
            img_size_val = int(img_size) if img_size and int(
                img_size) > 0 else None

            inference_engine.set_thresholds(conf_val, iou_val, img_size_val)

        elif choice == '6':
            new_video = input("Enter path to new video: ")
            if os.path.exists(new_video):
                video_path = new_video
                print(f"Video path updated to: {video_path}")
            else:
                print(f"Video file not found: {new_video}")

        elif choice == '7':
            print("\nCurrently loaded models:")
            for model_name, model in inference_engine.models.items():
                status = "Loaded" if model is not None else "Not loaded"
                path = inference_engine.model_paths[model_name] if model is not None else "None"
                print(f"- {model_name}: {status} ({path})")
            print(f"\nCurrent model: {inference_engine.current_model_name}")

        elif choice == '8':
            show_guide()

        elif choice == '9':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")


# Main function
def main():
    # Parse command-line arguments
    args = parse_args()

    # Initialize inference engine
    inference = ModelInference(
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        img_size=args.img_size
    )

    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load models based on mode
    if args.mode in ['single', 'interactive']:
        # Load the requested model for single mode
        if args.mode == 'single':
            inference.load_model(args.model, vars(args)[f'{args.model}_model'])
        else:
            # For interactive mode, try to load all models
            if os.path.exists(args.base_model):
                inference.load_model('base', args.base_model)

            if os.path.exists(args.enhanced_model):
                inference.load_model('enhanced', args.enhanced_model)

            if os.path.exists(args.quantized_model):
                inference.load_model('quantized', args.quantized_model)

    elif args.mode == 'compare':
        # Load all available models for comparison
        if os.path.exists(args.base_model):
            inference.load_model('base', args.base_model)
        else:
            print(f"Base model not found: {args.base_model}")

        if os.path.exists(args.enhanced_model):
            inference.load_model('enhanced', args.enhanced_model)
        else:
            print(f"Enhanced model not found: {args.enhanced_model}")

        if os.path.exists(args.quantized_model):
            inference.load_model('quantized', args.quantized_model)
        else:
            print(f"Quantized model not found: {args.quantized_model}")

    # Execute based on mode
    if args.mode == 'single':
        # Process video with selected model
        inference.process_video(
            video_path=args.video,
            output_path=args.output,
            display=args.display,
            max_frames=args.max_frames,
            start_frame=args.start_frame,
            verbose=args.verbose,
            save_detections=args.save_detections,
            detection_path=args.detection_path,
            rotate=args.rotate
        )

    elif args.mode == 'compare':
        # Compare all loaded models
        inference.compare_models(
            video_path=args.video,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            show_plots=args.show_plots,
            start_frame=args.start_frame,
            save_videos=args.save_videos,
            save_detections=args.save_detections,
            rotate=args.rotate
        )

    elif args.mode == 'interactive':
        # Run interactive mode
        interactive_mode(inference, args.video, args.output_dir)


if __name__ == "__main__":
    main()
