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

        # Detect headless
        self.headless = not bool(os.environ.get("DISPLAY"))
        print("Headless mode:", self.headless)

        # Load YOLO
        self.model = YOLO("yolov8n.pt")

        # Camera
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.source_fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_skip = max(1, self.source_fps // self.target_fps)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(
            f"{output_dir}/webcam_recording.mp4",
            fourcc, self.target_fps, (self.width, self.height)
        )

        self.person_distance_history = {}
        self.alert_cooldown = {}
        self.alerts_log = []
        self.frame_data = []

        self.focal_length = 800

        self.process = psutil.Process(os.getpid())

        self.frame_count = 0
        self.processed_frames = 0

    # --------------------------------------------------

    def estimate_distance(self, bbox):
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        if h == 0:
            return 999
        return max(0.5, (800 * 1.7) / h)

    # --------------------------------------------------

    def detect_distance_crossing(self, pid, distance, frame):
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
                if frame - self.alert_cooldown[pid] > 30:
                    self.alert_cooldown[pid] = frame
                    return t
        return None

    # --------------------------------------------------

    def log_alert(self, frame, distance, threshold):
        if threshold == 5:
            print(f"[FRAME {frame}] 🔴 CRITICAL — Person crossed 5m ({distance:.2f}m)")
        else:
            print(f"[FRAME {frame}] 🟠 WARNING — Person crossed 10m ({distance:.2f}m)")

        self.alerts_log.append({
            "frame": frame,
            "time_sec": frame / self.source_fps,
            "distance": round(distance, 2),
            "threshold": threshold
        })

    # --------------------------------------------------

    def process_frame(self, frame):
        results = self.model(frame, conf=0.5, classes=[0])

        if results[0].boxes is None:
            return frame

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        if len(boxes) > 0:
            print(f"[FRAME {self.frame_count}] 👤 Person detected: {len(boxes)}")

        for i, (box, conf) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = map(int, box)
            dist = self.estimate_distance(box)

            crossed = self.detect_distance_crossing(i, dist, self.frame_count)
            if crossed:
                self.log_alert(self.frame_count, dist, crossed)

            if dist <= 5:
                color = (0, 0, 255)
            elif dist <= 10:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{dist:.1f}m",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    # --------------------------------------------------

    def run(self):
        print("\n🚀 Person Distance Detector Running")
        print("Press CTRL+C to stop\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Camera read failed")
                    break

                self.frame_count += 1

                if (self.frame_count - 1) % self.frame_skip != 0:
                    continue

                self.processed_frames += 1

                frame = self.process_frame(frame)
                self.out.write(frame)

                if not self.headless:
                    cv2.imshow("Person Distance Detector", frame)
                    cv2.waitKey(1)
                else:
                    time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n🛑 CTRL+C pressed — shutting down...")

        self.cap.release()
        self.out.release()
        if not self.headless:
            cv2.destroyAllWindows()

        self.save_reports()

    # --------------------------------------------------

    def save_reports(self):
        with open(f"{self.output_dir}/alerts.json", "w") as f:
            json.dump(self.alerts_log, f, indent=2)

        print("\n✅ Files saved:")
        print(f"🎥 {self.output_dir}/webcam_recording.mp4")
        print(f"📄 {self.output_dir}/alerts.json")

# --------------------------------------------------

if __name__ == "__main__":
    detector = PersonDistanceDetector(camera_index=0, output_dir="output", target_fps=30)
    detector.run()
