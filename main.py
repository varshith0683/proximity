import cv2
import json
import threading
import queue
import time
import os
from ultralytics import YOLO
from collections import deque

class CaptureThread(threading.Thread):
    def __init__(self, cap, frame_queue):
        super().__init__(daemon=True)
        self.cap = cap
        self.frame_queue = frame_queue
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            try:
                self.frame_queue.put(frame, timeout=0.1)
            except queue.Full:
                pass

    def stop(self):
        self.running = False

class EvidenceWriter(threading.Thread):
    def __init__(self, fps, width, height, output_dir, pre_frames, post_frames):
        super().__init__(daemon=True)
        self.fps = fps
        self.width = width
        self.height = height
        self.output_dir = output_dir
        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.buffer = deque(maxlen=pre_frames)
        self.queue = queue.Queue()
        self.recording = False
        self.remaining = 0
        self.writer = None

    def push(self, frame, alert):
        self.queue.put((frame, alert))

    def run(self):
        while True:
            frame, alert = self.queue.get()
            if frame is None:
                break
            self.buffer.append(frame)
            if alert is not None and not self.recording:
                path = f"{self.output_dir}/evidence_{int(time.time())}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.writer = cv2.VideoWriter(path, fourcc, self.fps, (self.width, self.height))
                for bf in self.buffer:
                    self.writer.write(bf)
                self.recording = True
                self.remaining = self.post_frames
            if self.recording:
                self.writer.write(frame)
                self.remaining -= 1
                if self.remaining <= 0:
                    self.writer.release()
                    self.writer = None
                    self.recording = False

class PersonDistanceDetector:
    def __init__(self, camera_index=0, output_dir="output", detect_fps=10):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model = YOLO("yolov8n.pt")
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.detect_interval = 1.0 / detect_fps
        self.last_detect_time = 0.0
        self.frame_queue = queue.Queue(maxsize=300)
        self.capture_thread = CaptureThread(self.cap, self.frame_queue)
        self.evidence = EvidenceWriter(
            self.fps,
            self.width,
            self.height,
            output_dir,
            pre_frames=5 * self.fps,
            post_frames=5 * self.fps
        )
        self.tracks = {}
        self.track_id_counter = 0
        self.person_distance_history = {}
        self.alert_cooldown = {}
        self.alerts = []

    def estimate_distance(self, bbox):
        h = bbox[3] - bbox[1]
        if h <= 0:
            return 999
        return max(0.5, (800 * 1.7) / h)

    def iou(self, a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0

    def assign_ids(self, boxes):
        new_tracks = {}
        used = set()
        for tid, prev in self.tracks.items():
            best_iou = 0
            best_idx = -1
            for i, box in enumerate(boxes):
                if i in used:
                    continue
                v = self.iou(prev, box)
                if v > best_iou:
                    best_iou = v
                    best_idx = i
            if best_iou > 0.3:
                new_tracks[tid] = boxes[best_idx]
                used.add(best_idx)
        for i, box in enumerate(boxes):
            if i not in used:
                new_tracks[self.track_id_counter] = box
                self.track_id_counter += 1
        self.tracks = new_tracks
        return new_tracks

    def detect_cross(self, pid, dist, now):
        if pid not in self.person_distance_history:
            self.person_distance_history[pid] = deque(maxlen=10)
            self.alert_cooldown[pid] = 0.0
        hist = self.person_distance_history[pid]
        prev = hist[-1] if hist else 999
        hist.append(dist)
        for t in (5, 10):
            if prev > t and dist <= t and now - self.alert_cooldown[pid] > 1.0:
                self.alert_cooldown[pid] = now
                return t
        return None

    def run(self):
        self.capture_thread.start()
        self.evidence.start()
        start = time.time()
        try:
            while True:
                frame = self.frame_queue.get()
                now = time.time() - start
                alert = None
                if now - self.last_detect_time >= self.detect_interval:
                    self.last_detect_time = now
                    res = self.model(frame, conf=0.5, classes=[0])
                    boxes = res[0].boxes.xyxy.cpu().numpy() if res[0].boxes is not None else []
                    tracks = self.assign_ids(boxes)
                    for pid, box in tracks.items():
                        dist = self.estimate_distance(box)
                        crossed = self.detect_cross(pid, dist, now)
                        if crossed:
                            alert = crossed
                            self.alerts.append({
                                "time_sec": round(now, 2),
                                "distance": round(dist, 2),
                                "threshold": crossed
                            })
                        x1, y1, x2, y2 = map(int, box)
                        color = (0, 255, 0)
                        if dist <= 10:
                            color = (0, 165, 255)
                        if dist <= 5:
                            color = (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{dist:.1f}m", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                self.evidence.push(frame.copy(), alert)
        finally:
            self.capture_thread.stop()
            self.evidence.push(None, None)
            with open(f"{self.output_dir}/alerts.json", "w") as f:
                json.dump(self.alerts, f, indent=2)
            self.cap.release()

if __name__ == "__main__":
    detector = PersonDistanceDetector(camera_index=0, output_dir="output", detect_fps=10)
    detector.run()
