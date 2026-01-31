import cv2
import json
import threading
import queue
import time
import os
from ultralytics import YOLO
from collections import deque

class EvidenceWriter(threading.Thread):
    def __init__(self, frame_queue, fps, width, height, output_dir, pre_frames, post_frames):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.fps = fps
        self.width = width
        self.height = height
        self.output_dir = output_dir
        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.buffer = deque(maxlen=pre_frames)
        self.recording = False
        self.frames_remaining = 0
        self.writer = None

    def run(self):
        while True:
            item = self.frame_queue.get()
            if item is None:
                break
            frame, alert_label = item
            self.buffer.append(frame)
            if alert_label is not None and not self.recording:
                path = f"{self.output_dir}/evidence_{int(time.time())}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.writer = cv2.VideoWriter(path, fourcc, self.fps, (self.width, self.height))
                for bf in self.buffer:
                    self.writer.write(bf)
                self.recording = True
                self.frames_remaining = self.post_frames
            if self.recording:
                self.writer.write(frame)
                self.frames_remaining -= 1
                if self.frames_remaining <= 0:
                    self.writer.release()
                    self.writer = None
                    self.recording = False

class PersonDistanceDetector:
    def __init__(self, camera_index=0, output_dir="output", detect_fps=15):
        self.camera_index = camera_index
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model = YOLO("yolov8n.pt")
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.source_fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.detect_interval = 1.0 / detect_fps
        self.last_detect_time = 0.0
        self.tracks = {}
        self.track_id_counter = 0
        self.person_distance_history = {}
        self.alert_cooldown_time = {}
        self.alerts_log = []
        self.frame_queue = queue.Queue(maxsize=200)
        self.evidence_thread = EvidenceWriter(
            self.frame_queue,
            self.source_fps,
            self.width,
            self.height,
            self.output_dir,
            pre_frames=5 * self.source_fps,
            post_frames=5 * self.source_fps
        )
        self.evidence_thread.start()

    def estimate_distance(self, bbox):
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        if h <= 0:
            return 999
        return max(0.5, (800 * 1.7) / h)

    def iou(self, a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union else 0

    def assign_ids(self, boxes):
        new_tracks = {}
        used = set()
        for tid, prev in self.tracks.items():
            best_iou = 0
            best_idx = -1
            for i, box in enumerate(boxes):
                if i in used:
                    continue
                iou_score = self.iou(prev, box)
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_idx = i
            if best_iou > 0.3:
                new_tracks[tid] = boxes[best_idx]
                used.add(best_idx)
        for i, box in enumerate(boxes):
            if i not in used:
                new_tracks[self.track_id_counter] = box
                self.track_id_counter += 1
        self.tracks = new_tracks
        return self.tracks

    def detect_distance_crossing(self, pid, distance, now):
        if pid not in self.person_distance_history:
            self.person_distance_history[pid] = deque(maxlen=10)
            self.alert_cooldown_time[pid] = 0.0
        hist = self.person_distance_history[pid]
        prev = hist[-1] if hist else 999
        hist.append(distance)
        for t in [5, 10]:
            if (prev > t and distance <= t) or (prev == 999 and distance <= t):
                if now - self.alert_cooldown_time[pid] > 1.0:
                    self.alert_cooldown_time[pid] = now
                    return t
        return None

    def run(self):
        start_time = time.time()
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                now = time.time() - start_time
                alert_label = None
                if now - self.last_detect_time >= self.detect_interval:
                    self.last_detect_time = now
                    results = self.model(frame, conf=0.5, classes=[0])
                    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
                    tracks = self.assign_ids(boxes)
                    for pid, box in tracks.items():
                        dist = self.estimate_distance(box)
                        crossed = self.detect_distance_crossing(pid, dist, now)
                        if crossed:
                            alert_label = crossed
                            self.alerts_log.append({
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
                self.frame_queue.put((frame.copy(), alert_label))
        finally:
            self.cap.release()
            self.frame_queue.put(None)
            with open(f"{self.output_dir}/alerts.json", "w") as f:
                json.dump(self.alerts_log, f, indent=2)

if __name__ == "__main__":
    detector = PersonDistanceDetector(camera_index=0, output_dir="output", detect_fps=15)
    detector.run()
