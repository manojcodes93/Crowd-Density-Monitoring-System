import cv2
from app.detection import detect_people
from app.density import calculate_zones
from app.heatmap import generate_heatmap
from app.state import live_data

def run_engine():

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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
            continue

        frame = cv2.resize(frame, (640, 480))
        frame_count += 1

        if frame_count % frame_skip == 0:
            boxes, count = detect_people(frame)

        zones = calculate_zones(boxes, frame.shape)

        # Prediction
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

        # Update global state
        live_data["total"] = count
        live_data["zoneA"] = zones["A"]
        live_data["zoneB"] = zones["B"]
        live_data["zoneC"] = zones["C"]
        live_data["zoneD"] = zones["D"]
        live_data["prediction"] = predicted_value
        live_data["alert"] = alert