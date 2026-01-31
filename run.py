#For Windows Version use run.py to match the fps for evidence

import cv2
import json
import time
import os
from ultralytics import YOLO
from collections import deque
import signal

class PersonDistanceDetector:
    def __init__(self, camera_index=0, output_dir="output"):
        print("[INFO] System starting")

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model = YOLO("yolov8n.pt")
        print("[INFO] YOLO model loaded")

        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        if not self.cap.isOpened():
            raise RuntimeError("Camera could not be opened")

        print("[INFO] Camera opened")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.target_fps = 30
        self.frame_interval = 1.0 / self.target_fps

        self.recording_path = os.path.join(output_dir, "webcam_recording.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(
            self.recording_path,
            fourcc,
            self.target_fps,
            (self.width, self.height)
        )

        self.pre_event_seconds = 5
        self.post_event_seconds = 5
        self.buffer_size = int(self.pre_event_seconds * self.target_fps)
        self.frame_buffer = deque(maxlen=self.buffer_size)

        self.alerts = []
        self.last_distance = {}
        self.stop = False

        self.fps_log_interval = 2
        self.frame_count = 0
        self.fps_last_time = time.time()

        signal.signal(signal.SIGINT, self.shutdown)

    def shutdown(self, *args):
        print("\n[INFO] Shutdown signal received")
        self.stop = True

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
                frame_start = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    print("[WARN] Camera frame read failed")
                    break

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
                        elapsed = time.time() - start_time
                        print(f"[ALERT] Person crossed {crossed}m at {elapsed:.2f}s (distance={dist:.2f}m)")
                        self.alerts.append({
                            "time_sec": float(round(elapsed, 2)),
                            "distance": float(round(dist, 2)),
                            "threshold": crossed
                        })
                        active_event = crossed
                        post_event_end = elapsed + self.post_event_seconds

                self.frame_buffer.append(frame.copy())

                now = time.time()
                while last_written_time + self.frame_interval <= now:
                    self.out.write(frame)
                    last_written_time += self.frame_interval

                frame_end = time.time()
                self.frame_count += 1
                if frame_end - self.fps_last_time >= self.fps_log_interval:
                    fps = self.frame_count / (frame_end - self.fps_last_time)
                    print(f"[INFO] Approx. processing FPS: {fps:.2f}")
                    self.fps_last_time = frame_end
                    self.frame_count = 0

                if active_event and (time.time() - start_time) >= post_event_end:
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
        writer = cv2.VideoWriter(path, fourcc, self.target_fps, (self.width, self.height))

        for frame in self.frame_buffer:
            writer.write(frame)

        writer.release()
        self.frame_buffer.clear()
        print(f"[INFO] Evidence saved: {path}")

    def cleanup(self):
        print("[INFO] Releasing resources")
        self.cap.release()
        self.out.release()

        alerts_path = os.path.join(self.output_dir, "alerts.json")
        with open(alerts_path, "w") as f:
            json.dump(self.alerts, f, indent=2)

        print(f"[INFO] Alerts written to {alerts_path}")
        print("[INFO] System stopped cleanly")

if __name__ == "__main__":
    detector = PersonDistanceDetector(camera_index=0, output_dir="output")
    detector.run()
