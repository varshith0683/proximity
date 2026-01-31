import cv2
import json
import time
import os
from ultralytics import YOLO
from collections import deque
import signal

class PersonDistanceDetector:
    def __init__(self, camera_index=0, output_dir="output"):
        print("[INFO] System started")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model = YOLO("yolov8n.pt")

        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        if not self.cap.isOpened():
            raise RuntimeError("Camera could not be opened")
        print("[INFO] Camera activated")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.recording_path = os.path.join(output_dir, "webcam_recording.mp4")

        self.pre_event_seconds = 5
        self.post_event_seconds = 5
        self.frame_buffer = deque()

        self.alerts = []
        self.last_distance = {}
        self.stop = False

        signal.signal(signal.SIGINT, self.shutdown)

        self.measured_fps = self.measure_fps()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(
            self.recording_path,
            fourcc,
            self.measured_fps,
            (self.width, self.height)
        )
        self.frame_interval = 1.0 / self.measured_fps
        self.buffer_size = int(self.pre_event_seconds * self.measured_fps)
        self.frame_buffer = deque(maxlen=self.buffer_size)

    def shutdown(self, *args):
        print("[INFO] Shutdown signal received")
        self.stop = True

    def measure_fps(self, duration=3):
        start = time.time()
        frames = 0
        while time.time() - start < duration:
            ret, _ = self.cap.read()
            if ret:
                frames += 1
        elapsed = time.time() - start
        return max(frames / elapsed, 0.1)

    def estimate_distance(self, box):
        x1, y1, x2, y2 = box
        h = y2 - y1
        if h <= 0:
            return 999.0
        return float((800 * 1.7) / h)

    def detect_crossing(self, pid, dist):
        prev = self.last_distance.get(pid, 999.0)
        self.last_distance[pid] = dist
        if prev > 10 and dist <= 10:
            return 10
        if prev > 5 and dist <= 5:
            return 5
        return None

    def run(self):
        start_time = time.time()
        last_written_time = start_time
        active_event = None
        post_event_end = 0

        try:
            while not self.stop:
                ret, frame = self.cap.read()
                if not ret:
                    break

                now = time.time()
                elapsed = now - start_time

                results = self.model(frame, conf=0.5, classes=[0], verbose=False)
                boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    dist = self.estimate_distance(box)
                    crossed = self.detect_crossing(i, dist)

                    color = (0, 255, 0)
                    if dist <= 10:
                        color = (0, 165, 255)
                    if dist <= 5:
                        color = (0, 0, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{dist:.1f}m",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

                    if crossed:
                        print(f"[ALERT] Person crossed {crossed}m at {elapsed:.2f}s (distance={dist:.2f}m)")
                        self.alerts.append({
                            "time_sec": float(round(elapsed, 2)),
                            "distance": float(round(dist, 2)),
                            "threshold": crossed
                        })
                        active_event = crossed
                        post_event_end = elapsed + self.post_event_seconds

                self.frame_buffer.append(frame.copy())

                while last_written_time + self.frame_interval <= now:
                    self.out.write(frame)
                    last_written_time += self.frame_interval

                if active_event and elapsed >= post_event_end:
                    self.save_evidence()
                    active_event = None

        finally:
            self.cleanup()

    def save_evidence(self):
        if not self.frame_buffer:
            return

        filename = f"evidence_{int(time.time())}.mp4"
        path = os.path.join(self.output_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, self.measured_fps, (self.width, self.height))

        for frame in self.frame_buffer:
            writer.write(frame)

        writer.release()
        self.frame_buffer.clear()
        print(f"[INFO] Evidence captured: {path}")

    def cleanup(self):
        self.cap.release()
        self.out.release()

        alerts_path = os.path.join(self.output_dir, "alerts.json")
        with open(alerts_path, "w") as f:
            json.dump(self.alerts, f, indent=2)

if __name__ == "__main__":
    detector = PersonDistanceDetector(camera_index=0, output_dir="output")
    detector.run()
