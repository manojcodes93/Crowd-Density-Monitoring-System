import cv2
import time
import app.state as state

from app.detection import detect_people
from app.density import calculate_zones
from app.heatmap import generate_heatmap


def run_engine():
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
            time.sleep(0.1)
            continue

        frame = cv2.resize(frame, (640, 480))
        frame_count += 1

        # Run detection every few frames
        if frame_count % frame_skip == 0:
            boxes, count = detect_people(frame)

        # Generate heatmap
        heatmap_frame = generate_heatmap(frame, boxes)

        # 🔥 Update shared output frame
        with state.lock:
            state.output_frame = heatmap_frame.copy()

        # Zone calculation
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

        # 🔥 Update shared live data
        state.live_data["total"] = count
        state.live_data["zoneA"] = zones["A"]
        state.live_data["zoneB"] = zones["B"]
        state.live_data["zoneC"] = zones["C"]
        state.live_data["zoneD"] = zones["D"]
        state.live_data["prediction"] = predicted_value
        state.live_data["alert"] = alert