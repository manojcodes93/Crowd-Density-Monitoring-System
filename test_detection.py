import cv2
from app.detection import detect_people
from app.density import calculate_zones
from app.heatmap import generate_heatmap

cap = cv2.VideoCapture(0)

frame_skip = 3
frame_count = 0

boxes = []
count = 0

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

    frame = generate_heatmap(frame, boxes)
    cv2.imshow("Crowd Density Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
