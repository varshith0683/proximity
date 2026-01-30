import cv2
import numpy as np
import json
from ultralytics import YOLO
from collections import deque
from datetime import datetime
import os
import psutil
import time

class PersonDistanceDetector:
    def __init__(self, video_path, output_dir="output", target_fps=30):
        self.video_path = video_path
        self.output_dir = output_dir
        self.target_fps = target_fps
        os.makedirs(output_dir, exist_ok=True)
        
        # Load YOLO model
        self.model = YOLO('yolov8n.pt')
        self.cap = cv2.VideoCapture(video_path)
        
        # Video properties
        self.source_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Frame skip to achieve target FPS
        self.frame_skip = max(1, self.source_fps // self.target_fps)
        
        # Setup video writer for full video (at target FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(
            f"{output_dir}/output_video.mp4",
            fourcc, self.target_fps, (self.width, self.height)
        )
        
        # Alert tracking - detect when person crosses distance thresholds
        self.person_distance_history = {}  # person_id -> deque of (frame, distance)
        self.alerts_log = []
        self.frame_data = []
        self.frame_buffer = deque(maxlen=int(self.target_fps * 5))
        self.alert_frames = []
        
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
                return threshold
        
        return None
    
    def draw_distance_lines(self, frame):
        """Draw reference lines for 5m and 10m distances"""
        h = frame.shape[0]
        
        # 10 meter line (upper, further away)
        line_10m_y = int(h * 0.25)
        cv2.line(frame, (0, line_10m_y), (self.width, line_10m_y), (255, 0, 0), 2)
        cv2.putText(frame, "10m Line", (20, line_10m_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 5 meter line (lower, closer)
        line_5m_y = int(h * 0.5)
        cv2.line(frame, (0, line_5m_y), (self.width, line_5m_y), (0, 165, 255), 2)
        cv2.putText(frame, "5m Line", (20, line_5m_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    
    def get_performance_stats(self):
        """Get current CPU and RAM usage"""
        try:
            cpu_percent = self.process.cpu_percent(interval=0.01)
            ram_info = self.process.memory_info()
            ram_mb = ram_info.rss / 1024 / 1024
            return cpu_percent, ram_mb
        except:
            return 0, 0
    
    def process_video(self):
        """Main processing loop with performance profiling"""
        frame_count = 0
        processed_frames = 0
        start_time = time.time()
        
        print("="*70)
        print("STARTING VIDEO PROCESSING")
        print("="*70)
        print(f"Source FPS: {self.source_fps} | Target FPS: {self.target_fps} | Frame skip: {self.frame_skip}")
        print(f"Resolution: {self.width}x{self.height} | Total frames: {self.total_frames}")
        print("="*70)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames to achieve target FPS
            if (frame_count - 1) % self.frame_skip != 0:
                continue
            
            processed_frames += 1
            frame_start_time = time.time()
            
            # Get performance stats
            cpu_usage, ram_usage = self.get_performance_stats()
            
            # Run YOLO detection
            results = self.model(frame, conf=0.5, classes=[0])
            
            frame_data_entry = {
                "frame_number": frame_count,
                "processed_frame_number": processed_frames,
                "timestamp_seconds": float(frame_count / self.source_fps),
                "detections": [],
                "performance": {
                    "cpu_percent": round(cpu_usage, 2),
                    "ram_mb": round(ram_usage, 2)
                }
            }
            
            self.draw_distance_lines(frame)
            alert_triggered = False
            
            # Process detections
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for idx, (box, conf) in enumerate(zip(boxes, confidences)):
                    x1, y1, x2, y2 = map(int, box)
                    distance = self.estimate_distance(box)
                    
                    # Detect threshold crossing
                    crossing_threshold = self.detect_distance_crossing(idx, distance, frame_count)
                    
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
                        self.log_alert(frame_count, distance, crossing_threshold)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Display distance and alert
                    label = f"Person | Dist: {distance:.1f}m | {alert_level}"
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
            
            # Add frame counter and timestamp
            cv2.putText(frame, f"Frame: {frame_count}/{self.total_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Time: {frame_count/self.source_fps:.2f}s",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"CPU: {cpu_usage:.1f}% | RAM: {ram_usage:.1f}MB",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Add buffer frame
            self.frame_buffer.append({
                'frame': frame,
                'frame_num': frame_count,
                'timestamp': frame_count / self.source_fps,
                'alert': alert_triggered
            })
            
            # If alert triggered, store buffer frames
            if alert_triggered:
                self.alert_frames.extend(list(self.frame_buffer))
            
            # Write frame to output video
            self.out.write(frame)
            
            # Record processing time
            frame_process_time = time.time() - frame_start_time
            self.performance_stats["frame_processing_time"].append(frame_process_time)
            self.performance_stats["cpu_usage"].append(cpu_usage)
            self.performance_stats["ram_usage_mb"].append(ram_usage)
            
            print(f"Processing: {processed_frames} frames | CPU: {cpu_usage:.1f}% | RAM: {ram_usage:.1f}MB", end='\r')
        
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print("PROCESSING COMPLETE!")
        print("="*70)
        self.cap.release()
        self.out.release()
        
        return total_time
    
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
        print(f"\n🚨 ALERT: Person crossed {threshold}m line at frame {frame_num}")
    
    def create_evidence_videos(self):
        """Create evidence videos with 5 seconds before and after alert"""
        if not self.alerts_log:
            print("\nNo alerts detected. No evidence videos created.")
            return
        
        # Get unique alert frames
        unique_alert_frames = []
        seen_frames = set()
        
        for alert_data in self.alerts_log:
            alert_frame = alert_data['frame_number']
            if alert_frame not in seen_frames:
                unique_alert_frames.append(alert_frame)
                seen_frames.add(alert_frame)
        
        print(f"\nCreating {len(unique_alert_frames)} evidence videos...")
        
        # Create evidence video for each alert
        for idx, alert_frame in enumerate(unique_alert_frames):
            buffer_frames = int(self.source_fps * 5)
            start_frame = max(1, alert_frame - buffer_frames)
            end_frame = min(self.total_frames, alert_frame + buffer_frames)
            
            # Ensure minimum 7 seconds
            min_frames = int(self.source_fps * 7)
            current_length = end_frame - start_frame
            
            if current_length < min_frames:
                needed_frames = min_frames - current_length
                if start_frame > 1:
                    expand_back = min(needed_frames, start_frame - 1)
                    start_frame -= expand_back
                    needed_frames -= expand_back
                if needed_frames > 0 and end_frame < self.total_frames:
                    expand_forward = min(needed_frames, self.total_frames - end_frame)
                    end_frame += expand_forward
            
            self.extract_evidence_clip(start_frame, end_frame, idx + 1)
    
    def extract_evidence_clip(self, start_frame, end_frame, clip_num):
        """Extract and save evidence video clip"""
        self.cap = cv2.VideoCapture(self.video_path)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
        
        output_path = f"{self.output_dir}/evidence_clip_{clip_num}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.source_fps, (self.width, self.height))
        
        frame_num = start_frame
        
        while frame_num <= end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Run YOLO detection
            results = self.model(frame, conf=0.5, classes=[0])
            
            # Process detections
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    distance = self.estimate_distance(box)
                    
                    if distance <= 5:
                        color = (0, 0, 255)
                        alert_level = "CRITICAL (≤5m)"
                    elif distance <= 10:
                        color = (0, 165, 255)
                        alert_level = "WARNING (≤10m)"
                    else:
                        color = (0, 255, 0)
                        alert_level = f"Safe ({distance:.1f}m)"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"Person | Dist: {distance:.1f}m | {alert_level}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.putText(frame, f"Frame: {frame_num}/{self.total_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Time: {frame_num/self.source_fps:.2f}s",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if frame_num in [alert['frame_number'] for alert in self.alerts_log]:
                cv2.putText(frame, "*** ALERT FRAME ***", (10, self.height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            out.write(frame)
            frame_num += 1
        
        out.release()
        self.cap.release()
        print(f"✓ Evidence clip {clip_num} saved: {output_path}")
    
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
                "video_file": self.video_path,
                "total_frames": self.total_frames,
                "source_fps": self.source_fps,
                "target_fps": self.target_fps,
                "duration_seconds": float(self.total_frames / self.source_fps),
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
            "evidence_videos": [f"evidence_clip_{i+1}.mp4" for i in range(len(self.alerts_log))],
            "frame_by_frame_data": self.frame_data
        }
        
        output_path = f"{self.output_dir}/detection_report.json"
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ JSON report saved: {output_path}")
        return output_path
    
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
                "video_file": self.video_path,
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
    
    def run(self):
        """Run complete pipeline"""
        print("\n" + "="*70)
        print("PERSON DISTANCE DETECTOR - FULL PIPELINE")
        print("="*70)
        
        total_time = self.process_video()
        self.print_performance_summary(total_time)
        
        print("\nCreating evidence videos...")
        self.create_evidence_videos()
        
        self.save_alerts_only_json()
        self.save_json_report()
        
        print("\n" + "="*70)
        print("OUTPUT SUMMARY")
        print("="*70)
        print(f"✓ Output video    : {self.output_dir}/output_video.mp4")
        print(f"✓ Alerts JSON     : {self.output_dir}/alerts.json")
        print(f"✓ Full report     : {self.output_dir}/detection_report.json")
        print(f"✓ Total alerts    : {len(self.alerts_log)}")
        print(f"✓ Evidence clips  : {len(self.alerts_log)}")
        print("="*70 + "\n")


# Usage
if __name__ == "__main__":
    video_path = "data/test_2.mp4"
    output_dir = "output"
    
    detector = PersonDistanceDetector(video_path, output_dir=output_dir, target_fps=30)
    detector.run()