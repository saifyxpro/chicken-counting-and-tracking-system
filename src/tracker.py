"""
Chicken Tracking Module - AGGRESSIVE ANTI-FRAGMENTATION
ByteTrack with extreme settings to prevent ID switching

Problem: 433 unique IDs for 93 chickens = 4.6x fragmentation!
Solution: Much higher match threshold + longer buffer
"""

import numpy as np
import cv2
from typing import Dict, Set
import supervision as sv


class ChickenTracker:
    """
    ByteTrack tracker with AGGRESSIVE anti-fragmentation settings
    
    Key changes:
    - minimum_matching_threshold = 0.99 (almost always re-match)
    - lost_track_buffer = 900 (30 seconds!)
    - track_activation_threshold = 0.05 (track everything)
    """
    
    def __init__(
        self,
        track_thresh: float = 0.05,       # EXTREMELY LOW - track everything
        track_buffer: int = 900,           # 30 SECONDS at 30fps!
        match_thresh: float = 0.99,        # ALMOST 1.0 - always try to re-match
        min_frames_for_unique: int = 15    # Higher - must see 15 frames to count
    ):
        """
        Initialize ByteTrack with extreme anti-fragmentation
        """
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=30
        )
        
        self.track_frame_counts: Dict[int, int] = {}
        self.unique_ids: Set[int] = set()
        self.min_frames_for_unique = min_frames_for_unique
        self.frame_count = 0
        self.current_frame_count = 0
        
        # Store params for reset
        self._track_thresh = track_thresh
        self._track_buffer = track_buffer
        self._match_thresh = match_thresh
        
        print(f"✅ ByteTrack AGGRESSIVE: buffer={track_buffer}f, match={match_thresh}, min_frames={min_frames_for_unique}")
    
    def update(self, detections: list, frame: np.ndarray) -> sv.Detections:
        """Update tracker with new detections"""
        self.frame_count += 1
        
        if not detections:
            self.current_frame_count = 0
            return sv.Detections.empty()
        
        # Convert to supervision format
        xyxy = np.array([d["xyxy"] for d in detections])
        confidence = np.array([d["confidence"] for d in detections])
        class_id = np.array([d["class_id"] for d in detections])
        
        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )
        
        # Run tracker
        tracked = self.tracker.update_with_detections(sv_detections)
        
        # Update counts
        self.current_frame_count = len(tracked) if tracked.tracker_id is not None else 0
        
        if tracked.tracker_id is not None:
            for track_id in tracked.tracker_id:
                tid = int(track_id)
                
                if tid not in self.track_frame_counts:
                    self.track_frame_counts[tid] = 0
                self.track_frame_counts[tid] += 1
                
                # Stricter requirement: must be seen for min_frames
                if self.track_frame_counts[tid] >= self.min_frames_for_unique:
                    self.unique_ids.add(tid)
        
        return tracked
    
    def get_counts(self) -> dict:
        """Get current and total unique counts"""
        return {
            "current": self.current_frame_count,
            "unique": len(self.unique_ids),
            "frames": self.frame_count,
            "total_ids": len(self.track_frame_counts)  # Debug: all IDs ever created
        }
    
    def reset(self):
        """Reset tracker and counts"""
        self.tracker = sv.ByteTrack(
            track_activation_threshold=self._track_thresh,
            lost_track_buffer=self._track_buffer,
            minimum_matching_threshold=self._match_thresh,
            frame_rate=30
        )
        self.track_frame_counts.clear()
        self.unique_ids.clear()
        self.frame_count = 0
        self.current_frame_count = 0
        print("🔄 Tracker reset")
    
    def draw_tracks(self, frame: np.ndarray, tracked: sv.Detections) -> np.ndarray:
        """Draw tracked detections with ID numbers"""
        annotated = frame.copy()
        
        if tracked.tracker_id is None or len(tracked.tracker_id) == 0:
            return annotated
        
        for i, (xyxy, track_id) in enumerate(zip(tracked.xyxy, tracked.tracker_id)):
            x1, y1, x2, y2 = map(int, xyxy)
            tid = int(track_id)
            
            conf = tracked.confidence[i] if tracked.confidence is not None else 1.0
            
            colors = [
                (0, 255, 0), (255, 165, 0), (0, 255, 255),
                (255, 0, 255), (0, 165, 255), (255, 255, 0),
                (128, 0, 255), (255, 128, 0)
            ]
            color = colors[tid % len(colors)]
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            cx, cy = (x1 + x2) // 2, y1 - 15
            cv2.circle(annotated, (cx, cy), 18, color, -1)
            cv2.circle(annotated, (cx, cy), 18, (255, 255, 255), 2)
            
            label = str(tid)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(annotated, label, (cx - tw // 2, cy + th // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            cv2.putText(annotated, f"{conf:.0%}", (x1, y2 + 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated
    
    def draw_counts(self, frame: np.ndarray, current: int, unique: int) -> np.ndarray:
        """Overlay count info on frame"""
        annotated = frame.copy()
        
        cv2.rectangle(annotated, (10, 10), (280, 100), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, 10), (280, 100), (0, 255, 0), 2)
        
        cv2.putText(annotated, f"Current: {current}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f"Total Unique: {unique}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Debug info
        total_ids = len(self.track_frame_counts)
        if total_ids != unique:
            cv2.putText(annotated, f"(Raw IDs: {total_ids})", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        
        return annotated


if __name__ == "__main__":
    tracker = ChickenTracker()
    print("Tracker test passed!")
