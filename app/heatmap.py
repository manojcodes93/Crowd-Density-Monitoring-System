import numpy as np
import cv2

def generate_heatmap(frame, boxes):
    # Create blank heatmap (float)
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    # Add intensity at person centers
    for (x1, y1, x2, y2) in boxes:
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if 0 <= center_y < heatmap.shape[0] and 0 <= center_x < heatmap.shape[1]:
            heatmap[center_y, center_x] += 1

    # Blur to spread intensity
    heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)

    # Normalize to 0–255
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8
    heatmap = heatmap.astype(np.uint8)

    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay
    overlay = cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)

    return overlay
