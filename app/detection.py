from ultralytics import YOLO
import cv2

# Load YOLOv8 nano model (lightweight, fast)
model = YOLO("yolov8n.pt")

def detect_people(frame, confidence_threshold=0.4):

    results = model(frame, verbose=False)

    boxes = []
    count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])

            # Class 0 = person in COCO
            if cls == 0:
                conf = float(box.conf[0])
                if conf >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append((x1, y1, x2, y2))
                    count += 1

    return boxes, count