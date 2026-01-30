import cv2
import numpy as np
import json
from ultralytics import YOLO
from collections import deque
from datetime import datetime
import os
import psutil
import time
from threading import Thread

class PersonDistanceDetector:
    def __init__(self, output_dir="output", target_fps=30):
        self.camera_index = self._find_external_camera()
        self.output_dir = output_dir
        self.target_fps = target_fps
        os.makedirs(output_dir, exist_ok=True)
        
        # Load YOLO model
        self.model = YOLO('yolov8n.pt')
        self.cap = cv2.VideoCapture(self.camera_index)
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Video properties
        self.source_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if self.source_fps == 0:
            self.source_fps = 30  # Default FPS
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Frame skip to achieve target FPS
        self.frame_skip = max(1, self.source_fps // self.target_fps)
        
        # Setup video writer for recording
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(
            f"{output_dir}/webcam_recording.mp4",
            fourcc, self.target_fps, (self.width, self.height)
        )
        
        # Alert tracking
        self.person_distance_history = {}  # person_id -> deque of (frame, distance)
        self.alerts_log = []
        self.frame_data = []
        self.frame_buffer = deque(maxlen=int(self.target_fps * 5))
        
        # Camera calibration parameters
        self.focal_length = 800
        self.camera_height = 1.6
        
        # Performance monitoring
        self.performance_stats = {
            "cpu_usage": [],
            "ram_usage_mb": [],
            "frame_processing_time": []
        }
        self.process = psutil.Process(os.getpid())
        
        # Control flags
        self.running = True
        self.frame_count = 0
        self.processed_frames = 0
        
        # Alert history to avoid duplicate alerts
        self.alert_cooldown = {}  # person_id -> last_alert_frame
    
    def _find_external_camera(self):
        """
        Automatically find external USB camera.
        Skips index 0 (usually built-in laptop webcam) and finds first external camera.
        Falls back to index 0 if no external camera found.
        """
        print("\n" + "="*70)
        print("AUTO-DETECTING CAMERAS...")
        print("="*70)
        
        external_camera_found = False
        
        # Try indices 1-5 first (external USB cameras usually start at 1)
        for index in range(1, 6):
            print(f"Checking camera index {index}...", end=" ", flush=True)
            cap = cv2.VideoCapture(index)
            
            # Try to read a frame to verify camera works
            ret, frame = cap.read()
            
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                print(f"✓ FOUND - {width}x{height} @ {fps}fps")
                cap.release()
                
                print("="*70)
                print(f"✓ Using external camera at index {index}")
                print("="*70 + "\n")
                external_camera_found = True
                return index
            else:
                print("✗ Not available")
            
            cap.release()
        
        # Fallback to index 0 if no external camera found
        if not external_camera_found:
            print("\n" + "="*70)
            print("⚠️  No external camera detected.")
            print("Falling back to default camera (index 0)")
            print("="*70 + "\n")
            return 0
    
    def estimate_distance(self, bbox, person_height_pixels=100):
        """Estimate distance using person height in pixels and focal length"""
        x1, y1, x2, y2 = bbox
        height_pixels = y2 - y1
        
        if height_pixels == 0:
            return float('inf')
        
        avg_human_height = 1.7
        distance = (self.focal_length * avg_human_height) / height_pixels
        return max(0.5, distance)
    
    def detect_distance_crossing(self, person_id, distance, frame_count):
        """
        Detect when person crosses 5m or 10m thresholds
        Returns threshold value if crossing detected, None otherwise
        """
        thresholds = [10, 5]  # Check 10m first, then 5m
        
        if person_id not in self.person_distance_history:
            self.person_distance_history[person_id] = deque(maxlen=10)
            self.alert_cooldown[person_id] = -100
        
        history = self.person_distance_history[person_id]
        history.append((frame_count, distance))
        
        # Need at least 2 frames to detect crossing
        if len(history) < 2:
            return None
        
        prev_frame, prev_distance = list(history)[-2]
        curr_frame, curr_distance = list(history)[-1]
        
        # Check for crossing: previous > threshold AND current <= threshold
        for threshold in thresholds:
            if prev_distance > threshold and curr_distance <= threshold:
                # Add cooldown to avoid duplicate alerts
                if frame_count - self.alert_cooldown[person_id] > 30:  # 1 second cooldown
                    self.alert_cooldown[person_id] = frame_count
                    return threshold
        
        return None
    
    def display_alert(self, frame, alert_text, threshold):
        """Display alert in top right corner of frame"""
        if threshold == 5:
            color = (0, 0, 255)  # Red for CRITICAL
            alert_msg = "🚨 CRITICAL: 5m Line Crossed!"
        else:  # threshold == 10
            color = (0, 165, 255)  # Orange for WARNING
            alert_msg = "⚠️  WARNING: 10m Line Crossed!"
        
        # Create background box for alert
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Get text size
        text_size = cv2.getTextSize(alert_msg, font, font_scale, thickness)[0]
        text_width = text_size[0]
        text_height = text_size[1]
        
        # Position in top right
        x = self.width - text_width - 20
        y = 40
        
        # Draw background rectangle
        cv2.rectangle(frame, 
                     (x - 10, y - text_height - 10),
                     (x + text_width + 10, y + 10),
                     color, -1)
        
        # Draw border
        cv2.rectangle(frame,
                     (x - 10, y - text_height - 10),
                     (x + text_width + 10, y + 10),
                     (255, 255, 255), 2)
        
        # Put text
        cv2.putText(frame, alert_msg, (x, y),
                   font, font_scale, (255, 255, 255), thickness)
    
    def get_performance_stats(self):
        """Get current CPU and RAM usage"""
        try:
            cpu_percent = self.process.cpu_percent(interval=0.01)
            ram_info = self.process.memory_info()
            ram_mb = ram_info.rss / 1024 / 1024
            return cpu_percent, ram_mb
        except:
            return 0, 0
    
    def process_frame(self, frame):
        """Process a single frame"""
        frame_start_time = time.time()
        
        # Get performance stats
        cpu_usage, ram_usage = self.get_performance_stats()
        
        # Run YOLO detection
        results = self.model(frame, conf=0.5, classes=[0])
        
        frame_data_entry = {
            "frame_number": self.frame_count,
            "processed_frame_number": self.processed_frames,
            "timestamp_seconds": float(self.frame_count / self.source_fps),
            "detections": [],
            "performance": {
                "cpu_percent": round(cpu_usage, 2),
                "ram_mb": round(ram_usage, 2)
            }
        }
        
        alert_triggered = False
        alert_threshold = None
        
        # Process detections
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for idx, (box, conf) in enumerate(zip(boxes, confidences)):
                x1, y1, x2, y2 = map(int, box)
                distance = self.estimate_distance(box)
                
                # Detect threshold crossing
                crossing_threshold = self.detect_distance_crossing(idx, distance, self.frame_count)
                
                # Determine color and alert level
                if distance <= 5:
                    color = (0, 0, 255)
                    alert_level = "CRITICAL (≤5m)"
                elif distance <= 10:
                    color = (0, 165, 255)
                    alert_level = "WARNING (≤10m)"
                else:
                    color = (0, 255, 0)
                    alert_level = f"Safe ({distance:.1f}m)"
                
                # Log alert if threshold was crossed
                if crossing_threshold is not None:
                    alert_triggered = True
                    alert_threshold = crossing_threshold
                    self.log_alert(self.frame_count, distance, crossing_threshold)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Display distance info
                label = f"Dist: {distance:.1f}m | {alert_level}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Store frame data
                frame_data_entry["detections"].append({
                    "person_id": int(idx),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "distance_m": float(round(distance, 2)),
                    "alert_level": alert_level,
                    "threshold_crossed": crossing_threshold
                })
        
        self.frame_data.append(frame_data_entry)
        
        # Add performance stats on frame
        cv2.putText(frame, f"FPS: {self.target_fps} | CPU: {cpu_usage:.1f}% | RAM: {ram_usage:.1f}MB",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Display alert if triggered
        if alert_triggered:
            self.display_alert(frame, "", alert_threshold)
        
        # Record processing time
        frame_process_time = time.time() - frame_start_time
        self.performance_stats["frame_processing_time"].append(frame_process_time)
        self.performance_stats["cpu_usage"].append(cpu_usage)
        self.performance_stats["ram_usage_mb"].append(ram_usage)
        
        return frame
    
    def log_alert(self, frame_num, distance, threshold):
        """Log alert event when threshold is crossed"""
        alert_entry = {
            "frame_number": int(frame_num),
            "timestamp_seconds": float(frame_num / self.source_fps),
            "distance_m": float(round(distance, 2)),
            "threshold_crossed_m": int(threshold),
            "alert_type": f"Person crossed {threshold}m threshold"
        }
        self.alerts_log.append(alert_entry)
        print(f"🚨 ALERT: Person crossed {threshold}m line at frame {frame_num}")
    
    def run(self):
        """Run webcam processing with live display"""
        print("="*70)
        print("PERSON DISTANCE DETECTOR - WEBCAM LIVE FEED")
        print("="*70)
        print(f"Camera Index: {self.camera_index}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"Target FPS: {self.target_fps}")
        print("="*70)
        print("\nPress 'q' to quit the live feed and save results")
        print("Press 's' to save a screenshot\n")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to read frame from camera")
                break
            
            self.frame_count += 1
            
            # Skip frames to achieve target FPS
            if (self.frame_count - 1) % self.frame_skip != 0:
                continue
            
            self.processed_frames += 1
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Write to video file
            self.out.write(processed_frame)
            
            # Display live feed
            cv2.imshow('Person Distance Detector - Live Feed', processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n\nQuitting live feed...")
                self.running = False
                break
            elif key == ord('s'):
                screenshot_path = f"{self.output_dir}/screenshot_{self.frame_count}.png"
                cv2.imwrite(screenshot_path, processed_frame)
                print(f"Screenshot saved: {screenshot_path}")
        
        total_time = time.time() - start_time
        
        # Cleanup
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        
        self.print_performance_summary(total_time)
        self.save_alerts_only_json()
        self.save_json_report()
        
        print("\n" + "="*70)
        print("OUTPUT SUMMARY")
        print("="*70)
        print(f"✓ Webcam recording    : {self.output_dir}/webcam_recording.mp4")
        print(f"✓ Alerts JSON         : {self.output_dir}/alerts.json")
        print(f"✓ Full report         : {self.output_dir}/detection_report.json")
        print(f"✓ Total alerts        : {len(self.alerts_log)}")
        print(f"✓ Frames processed    : {self.processed_frames}")
        print(f"✓ Processing time     : {total_time:.2f}s")
        print("="*70 + "\n")
    
    def print_performance_summary(self, total_processing_time):
        """Print performance summary to console"""
        if not self.performance_stats["cpu_usage"]:
            return
        
        avg_cpu = np.mean(self.performance_stats["cpu_usage"])
        max_cpu = np.max(self.performance_stats["cpu_usage"])
        avg_ram = np.mean(self.performance_stats["ram_usage_mb"])
        max_ram = np.max(self.performance_stats["ram_usage_mb"])
        avg_frame_time = np.mean(self.performance_stats["frame_processing_time"])
        
        print("\n" + "="*70)
        print("PERFORMANCE PROFILING SUMMARY")
        print("="*70)
        print(f"CPU Usage       : Avg {avg_cpu:.1f}% | Max {max_cpu:.1f}%")
        print(f"RAM Usage       : Avg {avg_ram:.1f}MB | Max {max_ram:.1f}MB")
        print(f"Frame Time      : {avg_frame_time*1000:.2f}ms per frame")
        print(f"Total Time      : {total_processing_time:.2f}s")
        print("="*70)
    
    def save_alerts_only_json(self):
        """Save JSON file with only alerts data"""
        alerts_report = {
            "metadata": {
                "source": "Webcam",
                "generated_at": datetime.now().isoformat()
            },
            "alerts_summary": {
                "total_alerts": len(self.alerts_log),
                "alerts_5m": len([a for a in self.alerts_log if a['threshold_crossed_m'] == 5]),
                "alerts_10m": len([a for a in self.alerts_log if a['threshold_crossed_m'] == 10])
            },
            "alerts": self.alerts_log
        }
        
        output_path = f"{self.output_dir}/alerts.json"
        with open(output_path, 'w') as f:
            json.dump(alerts_report, f, indent=2)
        
        print(f"✓ Alerts JSON saved: {output_path}")
        return output_path
    
    def save_json_report(self):
        """Save comprehensive JSON report with alerts and performance data"""
        # Calculate performance statistics
        avg_cpu = np.mean(self.performance_stats["cpu_usage"]) if self.performance_stats["cpu_usage"] else 0
        max_cpu = np.max(self.performance_stats["cpu_usage"]) if self.performance_stats["cpu_usage"] else 0
        avg_ram = np.mean(self.performance_stats["ram_usage_mb"]) if self.performance_stats["ram_usage_mb"] else 0
        max_ram = np.max(self.performance_stats["ram_usage_mb"]) if self.performance_stats["ram_usage_mb"] else 0
        avg_frame_time = np.mean(self.performance_stats["frame_processing_time"]) if self.performance_stats["frame_processing_time"] else 0
        
        report = {
            "metadata": {
                "source": "Webcam",
                "camera_index": self.camera_index,
                "total_frames": self.frame_count,
                "processed_frames": self.processed_frames,
                "source_fps": self.source_fps,
                "target_fps": self.target_fps,
                "resolution": [self.width, self.height],
                "generated_at": datetime.now().isoformat()
            },
            "performance_profiling": {
                "cpu_usage": {
                    "average_percent": round(avg_cpu, 2),
                    "max_percent": round(max_cpu, 2)
                },
                "ram_usage": {
                    "average_mb": round(avg_ram, 2),
                    "max_mb": round(max_ram, 2)
                },
                "frame_processing": {
                    "average_time_ms": round(avg_frame_time * 1000, 2)
                }
            },
            "alerts_summary": {
                "total_alerts": len(self.alerts_log),
                "alerts_5m": len([a for a in self.alerts_log if a['threshold_crossed_m'] == 5]),
                "alerts_10m": len([a for a in self.alerts_log if a['threshold_crossed_m'] == 10])
            },
            "alerts_detailed": self.alerts_log,
            "frame_by_frame_data": self.frame_data
        }
        
        output_path = f"{self.output_dir}/detection_report.json"
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ JSON report saved: {output_path}")
        return output_path


# Usage
if __name__ == "__main__":
    # Automatically detects and uses external USB camera (Logitech, etc.)
    # Falls back to default camera if no external camera is found
    
    detector = PersonDistanceDetector(output_dir="output", target_fps=30)
    detector.run()