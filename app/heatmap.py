import numpy as np
import cv2

def generate_heatmap(frame, boxes):
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    for (x1, y1, x2, y2) in boxes:
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        heatmap[center_y, center_x] += 1

    # Blur to spread intensity
    heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)

    # Normalize to 0-255
    heatmap = np.clip(heatmap * 10, 0, 255).astype(np.uint8)

    # Apply color map
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay on original frame
    overlay = cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)

    return overlay
