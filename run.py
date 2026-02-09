import cv2
import json
import time
import os
import signal
import serial
from ultralytics import YOLO
from collections import deque


class PersonDistanceDetector:
    def __init__(self, camera_index=0, output_dir="output"):
        print("[SYSTEM] System started")

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        print("[MODEL] Loading YOLOv8 model...")
        self.model = YOLO("yolov8n.pt")
        print("[MODEL] YOLOv8 model loaded")

        print("[CAMERA] Connecting to camera")
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        if not self.cap.isOpened():
            raise RuntimeError("Camera could not be opened")

        print("[CAMERA] Camera connected")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[CAMERA] Stream resolution: {self.width}x{self.height}")

        self.serial_port = "/dev/ttySC1"
        self.baud_rate = 115200

        try:
            self.ser = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                timeout=0,
                write_timeout=0
            )
            print(f"[UART] Connected on {self.serial_port} @ {self.baud_rate}")
        except Exception as e:
            print(f"[UART] WARNING: UART not available ({e})")
            self.ser = None

        # self.recording_path = os.path.join(output_dir, "webcam_recording.mp4")
        # self.pre_event_seconds = 5
        # self.post_event_seconds = 5

        self.alerts = []
        self.last_distance = {}
        self.stop = False

        signal.signal(signal.SIGINT, self.shutdown)

        print("[SYSTEM] Measuring input FPS...")
        self.measured_fps = self.measure_fps()
        print(f"[SYSTEM] Measured FPS: {self.measured_fps:.2f}")

        # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # self.out = cv2.VideoWriter(
        #     self.recording_path,
        #     fourcc,
        #     self.measured_fps,
        #     (self.width, self.height)
        # )

        # self.frame_interval = 1.0 / self.measured_fps
        # self.buffer_size = int(self.pre_event_seconds * self.measured_fps)
        # self.frame_buffer = deque(maxlen=self.buffer_size)

    def shutdown(self, *args):
        print("[SYSTEM] Shutdown signal received")
        self.stop = True

    def measure_fps(self, duration=3):
        start = time.time()
        frames = 0
        while time.time() - start < duration:
            ret, _ = self.cap.read()
            if ret:
                frames += 1
        elapsed = time.time() - start
        return max(frames / elapsed, 1.0)

    def estimate_distance(self, box):
        x1, y1, x2, y2 = box
        h = y2 - y1
        if h <= 0:
            return 999.0
        return (800 * 1.7) / h

    def detect_crossing(self, pid, dist):
        prev = self.last_distance.get(pid, 999.0)
        self.last_distance[pid] = dist

        if prev > 8 and dist <= 8:
            return 8
        if prev > 4 and dist <= 4:
            return 4
        return None


    def send_uart_signal(self, value):
        if self.ser is None:
            return
        try:
            self.ser.write(f"{value}\n".encode("utf-8"))
            print(f"[UART] Sent signal: {value}")
        except Exception as e:
            print(f"[UART] WARNING: Write failed ({e})")

    def run(self):
        print("[SYSTEM] Processing started")

        start_time = time.time()
        # last_written_time = start_time
        # active_event = None
        # post_event_end = 0

        try:
            while not self.stop:
                ret, frame = self.cap.read()
                if not ret:
                    break

                print("[FRAME] Frame received")

                now = time.time()
                elapsed = now - start_time

                results = self.model(frame, conf=0.5, classes=[0], verbose=False)
                boxes = (
                    results[0].boxes.xyxy.cpu().numpy()
                    if results[0].boxes is not None
                    else []
                )

                for i, box in enumerate(boxes):
                    dist = self.estimate_distance(box)
                    crossed = self.detect_crossing(i, dist)

                    if crossed:
                        print(
                            f"[EVENT] Person crossed {crossed}m | "
                            f"Distance: {dist:.2f}m | "
                            f"Time: {elapsed:.2f}s"
                        )

                        if crossed == 4:
                            self.send_uart_signal(200)
                        elif crossed == 8:
                            self.send_uart_signal(100)


                        self.alerts.append({
                            "time_sec": round(elapsed, 2),
                            "distance": round(dist, 2),
                            "threshold": crossed
                        })

                        # active_event = crossed
                        # post_event_end = elapsed + self.post_event_seconds

                # self.frame_buffer.append(frame.copy())

                # while last_written_time + self.frame_interval <= now:
                #     self.out.write(frame)
                #     last_written_time += self.frame_interval

                # if active_event and elapsed >= post_event_end:
                #     self.save_evidence()
                #     active_event = None

        finally:
            self.cleanup()

    # def save_evidence(self):
    #     if not self.frame_buffer:
    #         return

    #     filename = f"evidence_{int(time.time())}.mp4"
    #     path = os.path.join(self.output_dir, filename)

    #     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #     writer = cv2.VideoWriter(
    #         path,
    #         fourcc,
    #         self.measured_fps,
    #         (self.width, self.height)
    #     )

    #     for frame in self.frame_buffer:
    #         writer.write(frame)

    #     writer.release()
    #     self.frame_buffer.clear()

    #     print(f"[EVIDENCE] Saved: {path}")

    def cleanup(self):
        print("[SYSTEM] Cleaning up resources")

        self.cap.release()
        # self.out.release()

        if self.ser:
            self.ser.close()

        alerts_path = os.path.join(self.output_dir, "alerts.json")
        with open(alerts_path, "w") as f:
            json.dump(self.alerts, f, indent=2)

        print("[SYSTEM] Shutdown complete")


if __name__ == "__main__":
    detector = PersonDistanceDetector(camera_index=0, output_dir="output")
    detector.run()
