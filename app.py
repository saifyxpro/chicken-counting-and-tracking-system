"""
Chicken Counting & Tracking System
Gradio Web Application - YOLO26s Edition

Features:
- YOLO26s detection (small-object optimized)
- Real-time chicken counting and tracking
- ByteTrack for consistent ID tracking
- CPU/NCNN inference support

Usage:
    python app.py
    Open: http://localhost:7860
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))

from src.detector import ChickenDetector
from src.tracker import ChickenTracker


MODEL_PATH = "models/chicken_yolov11s.pt"

detector = None
tracker = None
current_model_type = "yolov11s"


def init_models():
    """Initialize detector and tracker with YOLOv11s on CPU"""
    global detector, tracker, current_model_type
    
    if detector is None:
        print("Initializing YOLOv11s Chicken Model on CPU...")
        
        detector = ChickenDetector(
            model_path=MODEL_PATH,
            conf_threshold=0.35,
            iou_threshold=0.5,
            device="cpu",
            use_ncnn=False,
            imgsz=640
        )
        current_model_type = "yolov11s"
    
    if tracker is None:
        tracker = ChickenTracker()
    
    print("YOLOv11s Chicken Model ready!")
    return "Model: **YOLO26s (CPU)**"


def switch_model(use_ncnn: bool) -> str:
    """Switch between PyTorch and NCNN"""
    global detector, current_model_type
    
    detect_type = "NCNN (Fast CPU)" if use_ncnn else "PyTorch (Standard)"
    print(f"Switching model to: {detect_type}")
    
    detector = ChickenDetector(
        model_path=MODEL_PATH,
        conf_threshold=0.35,
        iou_threshold=0.5,
        device="cpu",
        use_ncnn=use_ncnn,
        imgsz=640
    )
    
    if use_ncnn:
        current_model_type = "yolov11s-ncnn"
        return "Using **YOLOv11s NCNN** (Fast CPU)"
    else:
        current_model_type = "yolov11s"
        return "Using **YOLOv11s** on **CPU**"


def process_image(image: np.ndarray) -> tuple:
    """Process image for chicken detection"""
    init_models()
    
    if image is None:
        return None, "No image provided"
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    
    detections = detector.detect(image_bgr)
    annotated = detector.draw_detections(image_bgr, detections)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    count = len(detections)
    summary = f"""
## Detection Results

| Metric | Count |
|--------|-------|
| Chickens Detected | **{count}** |

*For tracking with unique IDs, use video mode*
"""
    
    return annotated_rgb, summary


def process_video(video_path: str, progress=gr.Progress()) -> tuple:
    """Process video with tracking"""
    init_models()
    
    if video_path is None:
        return None, "No video provided"
    
    tracker.reset()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Could not open video"
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    temp_path = str(output_dir / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
    output_path = str(output_dir / f"tracked_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    
    frame_count = 0
    max_current = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % 10 == 0:
            progress(frame_count / total_frames, desc=f"Frame {frame_count}/{total_frames}")
        
        detections = detector.detect(frame)
        tracked = tracker.update(detections, frame)
        annotated = tracker.draw_tracks(frame, tracked)
        
        counts = tracker.get_counts()
        current = counts["current"]
        unique = counts["unique"]
        
        max_current = max(max_current, current)
        annotated = tracker.draw_counts(annotated, current, unique)
        
        cv2.putText(annotated, f"Frame: {frame_count}", (width - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(annotated)
    
    cap.release()
    out.release()
    
    try:
        import subprocess
        print(f"Converting video to H.264...")
        subprocess.run([
            'ffmpeg', '-y', '-i', temp_path,
            '-c:v', 'libx264', '-preset', 'fast',
            '-crf', '23', output_path
        ], check=True, capture_output=True)
        os.remove(temp_path)
        print(f"Video converted: {output_path}")
    except Exception as e:
        print(f"FFmpeg conversion failed: {e}")
        import shutil
        shutil.move(temp_path, output_path)
    
    final_counts = tracker.get_counts()
    
    summary = f"""
## Tracking Results

| Metric | Value |
|--------|-------|
| Total Frames | {total_frames} |
| Duration | {total_frames / fps:.1f}s |
| Max at Once | {max_current} |
| **Total Unique Chickens** | **{final_counts['unique']}** |

**Output:** `{output_path}`
"""
    
    return output_path, summary


def reset_tracker() -> str:
    """Reset the tracker"""
    global tracker
    if tracker:
        tracker.reset()
    return "Tracker reset! Ready for new video."


def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(
        title="Chicken Counting - YOLO26s",
        theme=gr.themes.Soft(primary_hue="orange")
    ) as demo:
        
        gr.Markdown("""
        # Chicken Counting & Tracking System
        
        **Powered by YOLO26s + ByteTrack** (Small Object Optimized)
        
        - Detect chickens with sequential numbering
        - Track and count unique chickens in videos
        - Optimized for CPU inference
        """)
        
        with gr.Tabs():
            with gr.TabItem("Image Detection"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(label="Upload Image", type="numpy", height=400)
                        image_btn = gr.Button("Detect Chickens", variant="primary", size="lg")
                    
                    with gr.Column():
                        image_output = gr.Image(label="Detection Result", height=400)
                
                image_summary = gr.Markdown(label="Results")
                
                image_btn.click(
                    fn=process_image,
                    inputs=[image_input],
                    outputs=[image_output, image_summary]
                )
            
            with gr.TabItem("Video Tracking"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="Upload Video", height=400)
                        with gr.Row():
                            video_btn = gr.Button("Process & Track", variant="primary", size="lg")
                            reset_btn = gr.Button("Reset Tracker", variant="secondary")
                    
                    with gr.Column():
                        video_output = gr.Video(label="Tracked Video", height=400)
                
                video_summary = gr.Markdown(label="Results")
                reset_status = gr.Markdown()
                
                video_btn.click(
                    fn=process_video,
                    inputs=[video_input],
                    outputs=[video_output, video_summary]
                )
                
                reset_btn.click(
                    fn=reset_tracker,
                    outputs=[reset_status]
                )
            
            with gr.TabItem("Settings"):
                gr.Markdown("### Model Settings")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
                        **Detection Backend**
                        - **PyTorch**: Standard inference
                        - **NCNN**: Faster CPU inference (43% speedup)
                        """)
                        
                        ncnn_toggle = gr.Checkbox(
                            label="Use NCNN (Fast CPU)",
                            value=False,
                            info="Enable for faster CPU inference"
                        )
                        
                        switch_btn = gr.Button("Apply Settings", variant="primary")
                        model_status = gr.Markdown("**Current:** YOLO26s (PyTorch CPU)")
                    
                    with gr.Column():
                        gr.Markdown("""
                        **Model Info**
                        
                        - Model: **YOLO26s** (Jan 2026)
                        - Optimized for: Small objects, edge devices
                        - Features: NMS-free, FP16 NCNN export
                        
                        See `train.md` for training guide.
                        """)
                
                switch_btn.click(
                    fn=switch_model,
                    inputs=[ncnn_toggle],
                    outputs=[model_status]
                )
        
        gr.Markdown("---\n**For best results, use steady camera footage with good lighting.**")
    
    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("Chicken Counting & Tracking System - YOLO26s")
    print("=" * 60)
    
    demo = create_interface()
    
    print("\nStarting Gradio server...")
    print("Open: http://localhost:7860\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
