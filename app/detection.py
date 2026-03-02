from ultralytics import YOLO
import torch
import cv2

# Load smallest model
model = YOLO("yolov8n.pt")

# Force CPU
device = "cpu"
model.to(device)


def detect_people(frame, confidence_threshold=0.4):


    results = model(
        frame,
        imgsz=640,
        conf=confidence_threshold,
        iou=0.45,
        device=device,
        verbose=False
    )

    boxes = []
    count = 0

    scale_x = frame.shape[1] / 640
    scale_y = frame.shape[0] / 480

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Scale back to original frame size
                x1 = int(x1 * scale_x)
                x2 = int(x2 * scale_x)
                y1 = int(y1 * scale_y)
                y2 = int(y2 * scale_y)

                boxes.append((x1, y1, x2, y2))
                count += 1

    return boxes, count