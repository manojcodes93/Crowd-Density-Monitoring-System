import cv2
import time
import threading

from app.detection import detect_people
from app.density import calculate_zones
from app.state import live_data
from app.heatmap import generate_heatmap

# Shared frame for video streaming
output_frame = None
lock = threading.Lock()


def run_engine():

    global output_frame

    print("Engine starting...")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Camera failed to open")
        return

    print("Camera opened successfully")

    frame_skip = 3
    frame_count = 0
    boxes = []
    count = 0

    recent_counts = []
    prediction_window = 5
    prediction_threshold = 10

    while True:
        ret, frame = cap.read()

        if not ret:
            print("⚠ Frame not received")
            time.sleep(0.1)
            continue

        # Resize frame
        frame = cv2.resize(frame, (640, 480))
        frame_count += 1

        # Run detection every N frames
        if frame_count % frame_skip == 0:
            boxes, count = detect_people(frame)

        # Calculate zone distribution
        zones = calculate_zones(boxes, frame.shape)

        # Generate heatmap overlay
        heatmap_frame = generate_heatmap(frame, boxes)

        # Store frame for streaming
        with lock:
            output_frame = heatmap_frame.copy()

        # ---------------- Prediction Logic ----------------
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

        # Update shared live state
        live_data["total"] = count
        live_data["zoneA"] = zones["A"]
        live_data["zoneB"] = zones["B"]
        live_data["zoneC"] = zones["C"]
        live_data["zoneD"] = zones["D"]
        live_data["prediction"] = predicted_value
        live_data["alert"] = alert