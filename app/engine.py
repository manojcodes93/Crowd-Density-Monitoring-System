import cv2
import threading
from datetime import datetime

from app.detection import detect_people
from app.density import calculate_zones

# Shared stats
current_stats = {
    "timestamp": None,
    "total": 0,
    "zones": {"A": 0, "B": 0, "C": 0, "D": 0},
    "predicted": 0,
    "alert": False
}

# Prediction memory
recent_counts = []
prediction_threshold = 10
prediction_window = 5


def start_engine():
    cap = cv2.VideoCapture(0)

    frame_skip = 3
    frame_count = 0

    global current_stats

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        frame_count += 1

        if frame_count % frame_skip == 0:
            boxes, count = detect_people(frame)

            zones = calculate_zones(boxes, frame.shape)

            # Prediction logic
            recent_counts.append(count)
            if len(recent_counts) > 10:
                recent_counts.pop(0)

            predicted_value = count
            alert = False

            if len(recent_counts) >= 2:
                growth_rate = (recent_counts[-1] - recent_counts[0]) / len(recent_counts)
                predicted_value = int(count + growth_rate * prediction_window)

                if predicted_value >= prediction_threshold:
                    alert = True

            current_stats = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total": count,
                "zones": zones,
                "predicted": predicted_value,
                "alert": alert
            }