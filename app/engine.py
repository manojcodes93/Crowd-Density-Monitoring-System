import cv2
import time
import numpy as np
import csv
import os
from datetime import datetime
import app.state as state
from app.detection import detect_people


def run_engine(source):

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Failed to open camera source: {source}")
        return

    # ---------- CREATE CSV HEADER IF NOT EXISTS ----------
    if not os.path.exists("crowd_log.csv"):
        with open("crowd_log.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp","total","zoneA","zoneB","zoneC","zoneD"])

    accumulated_heatmap = None
    recent_counts = []
    prediction_window = 5
    threshold = 5

    last_log_time = 0  # control 1-second logging

    state.engine_running = True

    while state.engine_running:

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        height, width, _ = frame.shape

        if accumulated_heatmap is None:
            accumulated_heatmap = np.zeros((height, width), dtype=np.float32)

        mid_x = width // 2
        mid_y = height // 2

        # ---------- DETECTION ----------
        boxes, count = detect_people(frame)

        zone_counts = {"A":0, "B":0, "C":0, "D":0}

        accumulated_heatmap *= 0.96

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

            radius = 50
            sigma = radius / 3

            y_grid, x_grid = np.ogrid[-radius:radius, -radius:radius]
            gaussian = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))

            x_start = max(center_x - radius, 0)
            x_end = min(center_x + radius, width)
            y_start = max(center_y - radius, 0)
            y_end = min(center_y + radius, height)

            heat_slice = accumulated_heatmap[y_start:y_end, x_start:x_end]

            g_x_start = max(0, radius - center_x)
            g_y_start = max(0, radius - center_y)
            g_x_end = g_x_start + (x_end - x_start)
            g_y_end = g_y_start + (y_end - y_start)

            heat_slice += gaussian[g_y_start:g_y_end, g_x_start:g_x_end] * 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        # ---------- PREDICTION ----------
        recent_counts.append(count)
        if len(recent_counts) > 10:
            recent_counts.pop(0)

        predicted = count
        alert_flag = False

        if len(recent_counts) >= 2:
            growth = (recent_counts[-1] - recent_counts[0]) / len(recent_counts)
            predicted = int(count + growth * prediction_window)
            if predicted >= threshold:
                alert_flag = True

        # ---------- HEATMAP ----------
        normalized = cv2.normalize(accumulated_heatmap, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)

        heat_color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.7, heat_color, 0.4, 0)

        cv2.line(overlay, (mid_x, 0), (mid_x, height), (255,255,255), 2)
        cv2.line(overlay, (0, mid_y), (width, mid_y), (255,255,255), 2)

        cv2.putText(overlay, f"People: {count}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.putText(overlay, f"Predicted: {predicted}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        if alert_flag:
            cv2.putText(overlay, "OVER CROWD ALERT!", (20,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        # ---------- UPDATE STATE ----------
        with state.lock:
            state.output_frame = overlay.copy()
            state.current_count = count
            state.zones = zone_counts
            state.prediction = predicted
            state.alert = alert_flag

        # ---------- LOG EVERY 1 SECOND ----------
        current_time = time.time()
        if current_time - last_log_time >= 1:
            timestamp = datetime.now().strftime("%H:%M:%S")
            with open("crowd_log.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    count,
                    zone_counts["A"],
                    zone_counts["B"],
                    zone_counts["C"],
                    zone_counts["D"]
                ])
            last_log_time = current_time

        time.sleep(0.03)

    cap.release()