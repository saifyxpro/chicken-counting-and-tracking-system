"""
Chicken Detection Module
Supports: YOLO26s (Ultralytics 2025/2026 - Small Object Optimized)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Ultralytics not installed! Run: pip install ultralytics>=8.3.0")


class ChickenDetector:
    """Chicken detector using YOLO26s (Small Object Optimized)"""
    
    def __init__(
        self,
        model_path: str = "yolo26s.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        device: str = "cpu",
        use_ncnn: bool = False,
        imgsz: int = 640
    ):
        """
        Initialize detector
        
        Args:
            model_path: Path to YOLO26 model (.pt or NCNN folder)
            conf_threshold: Confidence threshold (lower for small objects)
            iou_threshold: IoU threshold for NMS
            device: Device for inference (cpu/cuda:0)
            use_ncnn: Use NCNN format (CPU optimized)
            imgsz: Input image size (640 recommended for YOLO26)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = "cpu" if use_ncnn else device
        self.imgsz = imgsz
        self.model = None
        
        if YOLO_AVAILABLE:
            self._init_yolo(model_path, use_ncnn)
        else:
            raise RuntimeError("Ultralytics not installed. Cannot run detector.")
    
    def _init_yolo(self, model_path: str, use_ncnn: bool):
        """Initialize YOLO26 model with NCNN optimization support"""
        pt_path = Path(model_path)
        
        if use_ncnn:
            ncnn_folder = Path(str(pt_path).replace('.pt', '_ncnn_model'))
            if ncnn_folder.exists():
                model_to_load = str(ncnn_folder)
                print(f"Loading existing NCNN model: {model_to_load}")
            else:
                print(f"Exporting model to NCNN format (FP16 optimized)...")
                temp_model = YOLO(str(pt_path))
                try:
                    ncnn_path_exported = temp_model.export(format="ncnn", half=True, imgsz=self.imgsz)
                    model_to_load = ncnn_path_exported
                    print(f"Exported to NCNN: {model_to_load}")
                except Exception as e:
                    print(f"NCNN export failed: {e}")
                    print(f"Using PyTorch model on CPU instead")
                    model_to_load = str(pt_path)
        elif pt_path.exists():
            model_to_load = str(pt_path)
        else:
            model_to_load = "yolo26s.pt"
            print(f"Downloading {model_to_load} (pretrained COCO)")
        
        print(f"Loading YOLO26: {model_to_load}")
        self.model = YOLO(model_to_load)
        
        if not use_ncnn:
            self.model.to(self.device)
        
        print(f"YOLO26s ready on {self.device}")
    
    def detect(self, frame: np.ndarray) -> List[dict]:
        """Run detection on frame"""
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            verbose=False
        )[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": float(box.conf[0]),
                "class_id": class_id,
                "class_name": class_name,
                "xyxy": [x1, y1, x2, y2]
            })
        
        return detections
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[dict],
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """Draw bounding boxes with sequential numbering and corner stats"""
        annotated = frame.copy()
        height, width = frame.shape[:2]
        
        detections_sorted = sorted(detections, key=lambda d: d["bbox"][0])
        
        for i, det in enumerate(detections_sorted, 1):
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            label = f"#{i}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            
            cv2.rectangle(annotated, (x1, y1 - 24), (x1 + w + 10, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 6),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

        count = len(detections)
        overlay_text = f"Chickens: {count}"
        (tw, th), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        
        pad = 15
        bg_x1 = width - tw - pad * 2
        bg_y1 = pad
        bg_x2 = width - pad
        bg_y2 = pad + th + pad
        
        overlay = annotated.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
        
        cv2.putText(annotated, overlay_text, (bg_x1 + pad//2, bg_y2 - pad//2 - 2),
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2)
        
        return annotated


if __name__ == "__main__":
    try:
        detector = ChickenDetector()
        test = np.zeros((640, 640, 3), dtype=np.uint8)
        dets = detector.detect(test)
        print(f"Test detections: {len(dets)}")
    except Exception as e:
        print(f"Error: {e}")
