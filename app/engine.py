import cv2
import time
import numpy as np
from datetime import datetime
import app.state as state
from app.detection import detect_people
from app.database import insert_log


def run_engine(source):
    print("Engine started with source:", source)

    # ---------- CONFIG ----------
    prediction_window = 5
    density_threshold = 0.00005     # crowd density threshold
    sustained_seconds = 3           # alert after 3 seconds sustained

    recent_counts = []
    last_log_time = 0
    frame_count = 0
    last_boxes = []
    last_count = 0

    alert_start_time = None

    last_fps_time = time.time()
    fps = 0

    state.engine_running = True

    while state.engine_running:

        # ---------------- RECONNECT LOOP ----------------
        while True:
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if cap.isOpened():
                print("Camera connected")
                with state.lock:
                    state.camera_connected = True
                break
            else:
                print(f"Unable to connect to source: {source}. Retrying in 5 seconds...")
                with state.lock:
                    state.camera_connected = False
                    state.output_frame = None
                time.sleep(5)

        # ---------------- FRAME PROCESSING LOOP ----------------
        while state.engine_running and cap.isOpened():

            loop_start = time.time()

            if not cap.grab():
                print("Camera lost. Reconnecting...")
                with state.lock:
                    state.camera_connected = False
                cap.release()
                break

            ret, frame = cap.retrieve()
            if not ret:
                print("Frame retrieval failed. Possible corrupted stream.")
                with state.lock:
                    state.camera_connected = False
                cap.release()
                break

            frame = cv2.resize(frame, (640, 480))

            height, width, _ = frame.shape
            frame_area = width * height
            mid_x = width // 2
            mid_y = height // 2

            # ---------- FRAME SKIPPING ----------
            frame_count += 1
            if frame_count % 12 == 0:
                detect_start = time.time()
                last_boxes, last_count = detect_people(frame)
                processing_time = time.time() - detect_start
            else:
                processing_time = 0

            boxes = last_boxes
            count = last_count

            zone_counts = {"A": 0, "B": 0, "C": 0, "D": 0}

            for (x1, y1, x2, y2) in boxes:

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                if center_x < mid_x and center_y < mid_y:
                    zone_counts["A"] += 1
                elif center_x >= mid_x and center_y < mid_y:
                    zone_counts["B"] += 1
                elif center_x < mid_x and center_y >= mid_y:
                    zone_counts["C"] += 1
                else:
                    zone_counts["D"] += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            density = count / frame_area

            if density < density_threshold * 0.7:
                status = "NORMAL"
                status_color = (0, 255, 0)
            elif density < density_threshold:
                status = "MODERATE"
                status_color = (0, 255, 255)
            else:
                status = "CRITICAL"
                status_color = (0, 0, 255)

            alert_flag = False

            if density >= density_threshold:
                if alert_start_time is None:
                    alert_start_time = time.time()
                elif time.time() - alert_start_time >= sustained_seconds:
                    alert_flag = True
            else:
                alert_start_time = None

            recent_counts.append(count)
            if len(recent_counts) > 10:
                recent_counts.pop(0)

            predicted = count
            if len(recent_counts) >= 2:
                growth = (recent_counts[-1] - recent_counts[0]) / len(recent_counts)
                predicted = int(count + growth * prediction_window)

            cv2.line(frame, (mid_x, 0), (mid_x, height), (255, 255, 255), 1)
            cv2.line(frame, (0, mid_y), (width, mid_y), (255, 255, 255), 1)

            cv2.putText(frame, f"People: {count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.putText(frame, f"Predicted: {predicted}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(frame, f"Status: {status}",
                        (width - 200, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        status_color,
                        2)

            current_time = time.time()
            time_diff = current_time - last_fps_time

            if time_diff > 0:
                fps = 1 / time_diff
            else:
                fps = 0

            last_fps_time = current_time

            cv2.putText(frame, f"FPS: {int(fps)}",
                        (width - 120, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2)

            with state.lock:
                state.output_frame = frame.copy()
                state.current_count = count
                state.zones = zone_counts
                state.prediction = predicted
                state.alert = alert_flag
                state.status = status

            if current_time - last_log_time >= 1:
                timestamp = datetime.now().strftime("%H:%M:%S")
                insert_log(
                    timestamp,
                    count,
                    zone_counts["A"],
                    zone_counts["B"],
                    zone_counts["C"],
                    zone_counts["D"]
                )
                last_log_time = current_time