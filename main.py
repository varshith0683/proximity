import cv2
import numpy as np
import json
from ultralytics import YOLO
from collections import deque
import os
import time

class PersonDistanceDetector:
    def __init__(self, camera_index=0, output_dir="output", target_fps=15, pre_event_seconds=5, post_event_seconds=5):
        self.camera_index = camera_index
        self.output_dir = output_dir
        self.target_fps = target_fps
        self.pre_event_seconds = pre_event_seconds
        self.post_event_seconds = post_event_seconds
        os.makedirs(output_dir, exist_ok=True)

        self.headless = not bool(os.environ.get("DISPLAY"))

        self.model = YOLO("yolov8n.pt")

        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.frame_skip = max(1, self.source_fps // self.target_fps)

        self.out_path = f"{output_dir}/webcam_recording.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.out = cv2.VideoWriter(self.out_path, fourcc, self.target_fps, (self.width, self.height))

        self.tracks = {}
        self.track_id_counter = 0
        self.person_distance_history = {}
        self.alert_cooldown = {}
        self.alerts_log = []

        self.frame_buffer = deque(maxlen=self.pre_event_seconds * self.target_fps)
        self.processed_frame_count = 0

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
        area_a = (a[2]-a[0])*(a[3]-a[1])
        area_b = (b[2]-b[0])*(b[3]-b[1])
        union = area_a + area_b - inter
        return inter / union if union != 0 else 0

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
        prev = hist[-2] if len(hist) > 1 else distance
        for t in [5, 10]:
            if prev > t and distance <= t:
                if self.processed_frame_count - self.alert_cooldown[pid] > self.target_fps:
                    self.alert_cooldown[pid] = self.processed_frame_count
                    return t
        return None

    def log_alert(self, distance, threshold):
        self.alerts_log.append({
            "processed_frame": self.processed_frame_count,
            "time_sec": round(self.processed_frame_count / self.target_fps, 2),
            "distance": round(distance, 2),
            "threshold": threshold
        })

    def process_frame(self, frame):
        results = self.model(frame, conf=0.5, classes=[0])
        if results[0].boxes is None:
            return frame, False, []
        boxes = results[0].boxes.xyxy.cpu().numpy()
        tracks = self.assign_ids(boxes)
        alert_triggered = False
        alert_boxes = []
        for pid, box in tracks.items():
            x1, y1, x2, y2 = map(int, box)
            dist = self.estimate_distance(box)
            crossed = self.detect_distance_crossing(pid, dist)
            if crossed:
                self.log_alert(dist, crossed)
                alert_triggered = True
                alert_boxes.append((x1, y1, x2, y2, dist))
            color = (0, 0, 255) if dist <= 5 else (0, 165, 255) if dist <= 10 else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{dist:.1f}m", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame, alert_triggered, alert_boxes

    def save_evidence_clip(self, alert_boxes):
        frames = list(self.frame_buffer)
        frames += [frame for _ in range(self.post_event_seconds * self.target_fps)]
        w, h = self.width, self.height
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        idx = len(os.listdir(self.output_dir))
        out_path = f"{self.output_dir}/evidence_clip_{idx+1}.avi"
        out = cv2.VideoWriter(out_path, fourcc, self.target_fps, (w, h))
        for frame in frames:
            for x1, y1, x2, y2, dist in alert_boxes:
                color = (0, 0, 255) if dist <= 5 else (0, 165, 255) if dist <= 10 else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{dist:.1f}m", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(frame, "ALERT", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            out.write(frame)
        out.release()
        print(f"Evidence clip saved: {out_path}")

    def run(self):
        raw_count = 0
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                raw_count += 1
                if (raw_count-1) % self.frame_skip != 0:
                    continue
                self.processed_frame_count += 1
                self.frame_buffer.append(frame.copy())
                frame, alert_triggered, alert_boxes = self.process_frame(frame)
                self.out.write(frame)
                if alert_triggered:
                    self.save_evidence_clip(alert_boxes)
                if not self.headless:
                    cv2.imshow("Detector", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        except KeyboardInterrupt:
            pass
        self.cap.release()
        self.out.release()
        if not self.headless:
            cv2.destroyAllWindows()
        with open(f"{self.output_dir}/alerts.json", "w") as f:
            json.dump(self.alerts_log, f, indent=2)
        print(f"Alerts saved: {self.output_dir}/alerts.json ({len(self.alerts_log)} alerts)")

if __name__ == "__main__":
    detector = PersonDistanceDetector(camera_index=0, output_dir="output", target_fps=15)
    detector.run()
