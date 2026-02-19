import torch
import torchvision
import cv2
import numpy as np
from torchvision import transforms

# Loading pretrained Faster R-CNN (trained on COCO dataset)
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights
)

weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
model.eval()

# Convert image to PyTorch tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

def detect_people(frame, confidence_threshold=0.6):
    image = transform(frame) # Convert frame to tensor

    with torch.no_grad():
        outputs = model([image])[0] # Run detection (model expects list)

    boxes = []
    count = 0

    for i in range(len(outputs['boxes'])):
        label = outputs['labels'][i]
        score = outputs['scores'][i]
        
        # Class 1 = Person (COCO), filter by confidence
        if label == 1 and score > confidence_threshold:
            box = outputs['boxes'][i].detach().cpu().numpy().astype(int)
            boxes.append(box)  # Store bounding box
            count += 1         # Increase person count

    return boxes, count