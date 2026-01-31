import cv2
import numpy as np
import json
from ultralytics import YOLO
from collections import deque
import os

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
        self.recording_path = f"{output_dir}/webcam_recording.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.recording_path, fourcc, self.target_fps, (self.width, self.height))
        self.person_distance_history = {}
        self.alert_cooldown = {}
        self.alerts_log = []
        self.pre_event_seconds = 5
        self.post_event_seconds = 5
        self.processed_frame_count = 0
        self.track_id_counter = 0
        self.tracks = {}
        self.frame_buffer = deque(maxlen=self.target_fps * self.pre_event_seconds)

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
        if len(hist) == 0:
            hist.append(distance)
            for t in [5, 10]:
                if distance <= t:
                    if self.processed_frame_count - self.alert_cooldown[pid] > self.effective_fps:
                        self.alert_cooldown[pid] = self.processed_frame_count
                        return t
            return None
        hist.append(distance)
        prev, curr = hist[-2], hist[-1]
        for t in [5, 10]:
            if prev > t and curr <= t:
                if self.processed_frame_count - self.alert_cooldown[pid] > self.effective_fps:
                    self.alert_cooldown[pid] = self.processed_frame_count
                    return t
        return None

    def log_alert(self, distance, threshold):
        self.alerts_log.append({
            "processed_frame": self.processed_frame_count,
            "time_sec": round(self.processed_frame_count / self.effective_fps, 2),
            "distance": round(distance, 2),
            "threshold": threshold
        })

    def process_frame(self, frame):
        results = self.model(frame, conf=0.5, classes=[0])
        if results[0].boxes is None:
            return frame, False
        boxes = results[0].boxes.xyxy.cpu().numpy()
        tracks = self.assign_ids(boxes)
        alert_triggered = False
        for pid, box in tracks.items():
            x1, y1, x2, y2 = map(int, box)
            dist = self.estimate_distance(box)
            crossed = self.detect_distance_crossing(pid, dist)
            if crossed:
                self.log_alert(dist, crossed)
                alert_triggered = True
            if dist <= 5:
                color = (0, 0, 255)
            elif dist <= 10:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{dist:.1f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame, alert_triggered

    def save_evidence_clip(self, alert_frame_count):
        fps = self.effective_fps
        w, h = self.width, self.height
        start_frame = max(0, alert_frame_count - fps * self.pre_event_seconds)
        end_frame = alert_frame_count + fps * self.post_event_seconds
        output_path = f"{self.output_dir}/evidence_clip_{alert_frame_count}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        cap = cv2.VideoCapture(self.recording_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current = start_frame
        alert_frame_set = {alert_frame_count}
        while current <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame, conf=0.5, classes=[0])
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    dist = self.estimate_distance(box)
                    if dist <= 5:
                        color = (0, 0, 255)
                    elif dist <= 10:
                        color = (0, 165, 255)
                    else:
                        color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{dist:.1f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if current in alert_frame_set:
                cv2.putText(frame, "ALERT", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out.write(frame)
            current += 1
        out.release()
        cap.release()
        print(f"Evidence clip saved: {output_path}")

    def run(self):
        raw_frame_count = 0
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                raw_frame_count += 1
                if (raw_frame_count - 1) % self.frame_skip != 0:
                    continue
                self.processed_frame_count += 1
                frame, alert_triggered = self.process_frame(frame)
                self.out.write(frame)
                self.frame_buffer.append(frame.copy())
                if alert_triggered:
                    self.save_evidence_clip(self.processed_frame_count)
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
        print(f"Alerts saved: {self.output_dir}/alerts.json ({len(self.alerts_log)} alerts)")

if __name__ == "__main__":
    detector = PersonDistanceDetector(camera_index=0, output_dir="output", target_fps=30)
    detector.run()
