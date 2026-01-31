import cv2
import numpy as np
import json
import threading
import time
import os
from ultralytics import YOLO
from collections import deque
from datetime import datetime
from queue import Queue, Empty


class CaptureThread(threading.Thread):
    def __init__(self, camera_index, width, height, fps):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.buffer = deque(maxlen=fps * 6)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.frame_index = 0
        self.latest_frame = None
        self.has_new_frame = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_index += 1
            with self.lock:
                self.buffer.append((self.frame_index, frame))
                self.latest_frame = (self.frame_index, frame)
            self.has_new_frame.set()

    def get_latest(self):
        self.has_new_frame.wait(timeout=1.0)
        self.has_new_frame.clear()
        with self.lock:
            return self.latest_frame

    def get_buffer_snapshot(self):
        with self.lock:
            return list(self.buffer)

    def stop(self):
        self.stop_event.set()
        self.join(timeout=2.0)
        self.cap.release()


class EvidenceWriter(threading.Thread):
    def __init__(self, capture_thread, output_dir, fps, width, height):
        super().__init__(daemon=True)
        self.capture_thread = capture_thread
        self.output_dir = output_dir
        self.fps = fps
        self.width = width
        self.height = height

        self.trigger_event = threading.Event()
        self.trigger_frame_index = 0
        self.stop_event = threading.Event()
        self.clip_index = 0
        self.saved_files = []

        self.pre_seconds = 5
        self.post_seconds = 5

    def trigger(self, frame_index):
        self.trigger_frame_index = frame_index
        self.trigger_event.set()

    def run(self):
        while not self.stop_event.is_set():
            if not self.trigger_event.wait(timeout=0.5):
                continue
            self.trigger_event.clear()
            if self.stop_event.is_set():
                break
            self._write_clip()

    def _write_clip(self):
        self.clip_index += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{self.output_dir}/evidence_clip_{ts}_{self.clip_index}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        pre_frames_needed = self.fps * self.pre_seconds
        post_frames_needed = self.fps * self.post_seconds
        alert_idx = self.trigger_frame_index

        snapshot = self.capture_thread.get_buffer_snapshot()

        pre_frames = []
        for idx, frame in snapshot:
            if idx <= alert_idx:
                pre_frames.append((idx, frame))

        if len(pre_frames) > pre_frames_needed:
            pre_frames = pre_frames[-pre_frames_needed:]

        for idx, frame in pre_frames:
            writer.write(frame)

        post_written = 0
        last_written_idx = alert_idx
        while post_written < post_frames_needed and not self.stop_event.is_set():
            snapshot = self.capture_thread.get_buffer_snapshot()
            for idx, frame in snapshot:
                if idx <= last_written_idx:
                    continue
                writer.write(frame)
                last_written_idx = idx
                post_written += 1
                if post_written >= post_frames_needed:
                    break
            if post_written < post_frames_needed:
                time.sleep(0.05)

        writer.release()
        self.saved_files.append(output_path)
        print(f"Evidence clip saved: {output_path}")

    def stop(self):
        self.stop_event.set()
        self.trigger_event.set()
        self.join(timeout=3.0)


class PersonDistanceDetector:
    def __init__(self, camera_index=0, output_dir="output", target_fps=30):
        self.output_dir = output_dir
        self.target_fps = target_fps
        os.makedirs(output_dir, exist_ok=True)

        self.headless = not bool(os.environ.get("DISPLAY"))

        self.model = YOLO("yolov8n.pt")

        probe = cv2.VideoCapture(camera_index)
        probe.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        probe.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        probe.release()

        self.capture_thread = CaptureThread(camera_index, self.width, self.height, target_fps)

        self.evidence_writer = EvidenceWriter(
            self.capture_thread, output_dir, target_fps, self.width, self.height
        )

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(
            f"{output_dir}/webcam_recording.mp4",
            fourcc, target_fps, (self.width, self.height)
        )

        self.person_distance_history = {}
        self.alert_cooldown = {}
        self.alerts_log = []

        self.processed_frame_count = 0
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

        if len(hist) == 0:
            hist.append(distance)
            for t in [5, 10]:
                if distance <= t:
                    if self.processed_frame_count - self.alert_cooldown[pid] > self.target_fps:
                        self.alert_cooldown[pid] = self.processed_frame_count
                        return t
            return None

        hist.append(distance)
        prev, curr = hist[-2], hist[-1]

        for t in [5, 10]:
            if prev > t and curr <= t:
                if self.processed_frame_count - self.alert_cooldown[pid] > self.target_fps:
                    self.alert_cooldown[pid] = self.processed_frame_count
                    return t
        return None

    def log_alert(self, distance, threshold, capture_frame_index):
        self.alerts_log.append({
            "processed_frame": self.processed_frame_count,
            "capture_frame_index": capture_frame_index,
            "time_sec": round(self.processed_frame_count / self.target_fps, 2),
            "distance": round(distance, 2),
            "threshold": threshold
        })

    def process_frame(self, frame, capture_frame_index):
        results = self.model(frame, conf=0.5, classes=[0])

        if results[0].boxes is None:
            return frame, False

        boxes = results[0].boxes.xyxy.cpu().numpy()
        tracks = self.assign_ids(boxes)
        alert_triggered = False
        alert_capture_idx = capture_frame_index

        for pid, box in tracks.items():
            x1, y1, x2, y2 = map(int, box)
            dist = self.estimate_distance(box)
            crossed = self.detect_distance_crossing(pid, dist)
            if crossed:
                self.log_alert(dist, crossed, capture_frame_index)
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

    def run(self):
        self.capture_thread.start()
        self.evidence_writer.start()

        last_capture_idx = 0

        try:
            while True:
                result = self.capture_thread.get_latest()
                if result is None:
                    continue

                capture_idx, raw_frame = result
                if capture_idx == last_capture_idx:
                    continue
                last_capture_idx = capture_idx

                self.processed_frame_count += 1
                frame = raw_frame.copy()
                frame, alert_triggered = self.process_frame(frame, capture_idx)
                self.out.write(frame)

                if alert_triggered:
                    self.evidence_writer.trigger(capture_idx)

                if not self.headless:
                    cv2.imshow("Detector", frame)
                    cv2.waitKey(1)

        except KeyboardInterrupt:
            pass

        self.evidence_writer.stop()
        self.capture_thread.stop()
        self.out.release()

        if not self.headless:
            cv2.destroyAllWindows()

        with open(f"{self.output_dir}/alerts.json", "w") as f:
            json.dump(self.alerts_log, f, indent=2)
        print(f"Alerts saved: {self.output_dir}/alerts.json ({len(self.alerts_log)} alerts)")

        if self.evidence_writer.saved_files:
            print(f"Evidence clips saved: {len(self.evidence_writer.saved_files)}")
            for f in self.evidence_writer.saved_files:
                print(f"  {f}")
        else:
            print("No evidence clips recorded.")


if __name__ == "__main__":
    detector = PersonDistanceDetector(camera_index=0, output_dir="output", target_fps=30)
    detector.run()