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
    def __init__(self, camera_index=0, output_dir="output", target_fps=30):
        self.camera_index = camera_index
        self.output_dir = output_dir
        self.target_fps = target_fps
        os.makedirs(output_dir, exist_ok=True)

        self.headless = not bool(os.environ.get("DISPLAY"))

        self.model = YOLO("yolov8n.pt")

        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.source_fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_skip = max(1, self.source_fps // self.target_fps)
        self.effective_fps = self.target_fps

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(
            f"{output_dir}/webcam_recording.mp4",
            fourcc, self.target_fps, (self.width, self.height)
        )

        self.person_distance_history = {}
        self.alert_cooldown = {}
        self.alerts_log = []

        self.pre_event_seconds = 5
        self.post_event_seconds = 5

        self.frame_buffer = deque(maxlen=self.effective_fps * self.pre_event_seconds)
        self.event_active = False
        self.event_writer = None
        self.event_frames_remaining = 0
        self.event_index = 0
        self.event_latch = 0

        self.frame_count = 0
        self.track_id_counter = 0
        self.tracks = {}

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
        if union == 0:
            return 0
        return inter / union

    def assign_ids(self, boxes):
        new_tracks = {}
        used = set()

        for tid, prev_box in self.tracks.items():
            best_iou = 0
            best_idx = -1
            for i, box in enumerate(boxes):
                if i in used:
                    continue
                score = self.iou(prev_box, box)
                if score > best_iou:
                    best_iou = score
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

    def detect_distance_crossing(self, pid, distance):
        if pid not in self.person_distance_history:
            self.person_distance_history[pid] = deque(maxlen=10)
            self.alert_cooldown[pid] = -100

        hist = self.person_distance_history[pid]
        hist.append(distance)

        if len(hist) < 2:
            return None

        prev, curr = hist[-2], hist[-1]

        for t in [10, 5]:
            if prev > t and curr <= t:
                if self.frame_count - self.alert_cooldown[pid] > self.effective_fps:
                    self.alert_cooldown[pid] = self.frame_count
                    return t
        return None

    def log_alert(self, distance, threshold):
        self.alerts_log.append({
            "time_sec": self.frame_count / self.effective_fps,
            "distance": round(distance, 2),
            "threshold": threshold
        })

    def handle_event_recording(self, frame, event_triggered):
        if event_triggered and not self.event_active:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/evidence_{ts}_{self.event_index}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.event_writer = cv2.VideoWriter(filename, fourcc, self.target_fps, (self.width, self.height))
            for f in self.frame_buffer:
                self.event_writer.write(f)
            self.event_frames_remaining = self.effective_fps * self.post_event_seconds
            self.event_active = True
            self.event_index += 1

        if self.event_active:
            self.event_writer.write(frame)
            self.event_frames_remaining -= 1
            if self.event_frames_remaining <= 0:
                self.event_writer.release()
                self.event_active = False

    def process_frame(self, frame):
        event_triggered = False
        results = self.model(frame, conf=0.5, classes=[0])

        if results[0].boxes is None:
            return frame, False

        boxes = results[0].boxes.xyxy.cpu().numpy()
        tracks = self.assign_ids(boxes)

        for pid, box in tracks.items():
            x1, y1, x2, y2 = map(int, box)
            dist = self.estimate_distance(box)
            crossed = self.detect_distance_crossing(pid, dist)
            if crossed:
                self.log_alert(dist, crossed)
                self.event_latch = self.effective_fps
            if dist <= 5:
                color = (0, 0, 255)
            elif dist <= 10:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{dist:.1f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if self.event_latch > 0:
            event_triggered = True
            self.event_latch -= 1

        return frame, event_triggered

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                self.frame_count += 1
                if (self.frame_count - 1) % self.frame_skip != 0:
                    continue

                self.frame_buffer.append(frame.copy())
                frame, event_triggered = self.process_frame(frame)
                self.out.write(frame)
                self.handle_event_recording(frame, event_triggered)

                if not self.headless:
                    cv2.imshow("Detector", frame)
                    cv2.waitKey(1)

        except KeyboardInterrupt:
            pass

        self.cap.release()
        self.out.release()
        if not self.headless:
            cv2.destroyAllWindows()

        with open(f"{self.output_dir}/alerts.json", "w") as f:
            json.dump(self.alerts_log, f, indent=2)

if __name__ == "__main__":
    detector = PersonDistanceDetector(camera_index=0, output_dir="output", target_fps=30)
    detector.run()
