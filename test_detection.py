import cv2
import os
from app.detection import detect_people
from app.density import calculate_zones
from app.heatmap import generate_heatmap
import csv
from datetime import datetime

cap = cv2.VideoCapture(0)

frame_skip = 3
frame_count = 0

boxes = []
count = 0


file_exists = os.path.isfile("crowd_log.csv")

file = open("crowd_log.csv", mode="a", newline="")
writer = csv.writer(file)

if not file_exists:
    writer.writerow(["timestamp", "total", "A", "B", "C", "D"])

last_logged_time = None

recent_counts = []
prediction_threshold = 10
prediction_window = 5

predicted_value = 0
prediction_alert = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster inference
    frame = cv2.resize(frame, (640, 480))

    frame_count += 1

    # Run detection every 3rd frame
    if frame_count % frame_skip == 0:
        boxes, count = detect_people(frame)

    # Draw bounding boxes
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Calculate zone counts
    zones = calculate_zones(boxes, frame.shape)

    THRESHOLD = 5
    alerts = {
        zone: zones[zone] > THRESHOLD
        for zone in zones
    }

    current_time = datetime.now()

    if last_logged_time is None or (current_time - last_logged_time).seconds >= 1:
        writer.writerow([
            current_time.strftime("%Y-%m-%d %H:%M:%S"),
            count,
            zones["A"],
            zones["B"],
            zones["C"],
            zones["D"]
        ])
        recent_counts.append(count)

        if len(recent_counts) > 10:
            recent_counts.pop(0)
        
        last_logged_time = current_time

    prediction_alert = False
    predicted_value = count

    if len(recent_counts) >= 2:
        growth_rate = (recent_counts[-1] - recent_counts[0]) / len(recent_counts)
        predicted_value = int(count + growth_rate * prediction_window)

        if predicted_value >= prediction_threshold:
            prediction_alert = True

    # Draw zone dividing lines
    h, w, _ = frame.shape
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)
    cv2.line(frame, (0, h // 2), (w, h // 2), (255, 255, 255), 2)

    # Display total count
    cv2.putText(frame, f"Total Count: {count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2)

    start_y = 80

    for i, zone in enumerate(zones):
        text = f"{zone}: {zones[zone]}"

        if alerts[zone]:
            text += "  ALERT!"
            color = (0, 0, 255)  # Red
        else:
            color = (255, 0, 0)  # Blue

        cv2.putText(frame, text,
                    (20, start_y + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2)
    
    cv2.putText(frame, f"Predicted (5s): {predicted_value}",
                (20, start_y + 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2)
    
    if prediction_alert:
        cv2.putText(frame, "⚠ Overcrowding Likely!",
                    (20, start_y + 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2)

    frame = generate_heatmap(frame, boxes)
    cv2.imshow("Crowd Density Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

file.close()
cap.release()
cv2.destroyAllWindows()
