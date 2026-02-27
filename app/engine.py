import cv2
import time
import app.state as state
from app.detection import detect_people

def run_engine():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        boxes, count = detect_people(frame)

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.putText(frame, f"People: {count}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        with state.lock:
            state.output_frame = frame.copy()

        time.sleep(0.03)