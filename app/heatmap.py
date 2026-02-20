import numpy as np
import cv2

# Global heatmap memory
accumulated_heatmap = None

def generate_heatmap(frame, boxes):
    global accumulated_heatmap

    h, w, _ = frame.shape

    # Initialize heatmap once
    if accumulated_heatmap is None:
        accumulated_heatmap = np.zeros((h, w), dtype=np.float32)

    # Decay old heat (controls memory length)
    decay_factor = 0.95
    accumulated_heatmap *= decay_factor

    # Add new Gaussian blobs
    for (x1, y1, x2, y2) in boxes:
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        radius = 40

        x_min = max(center_x - radius, 0)
        x_max = min(center_x + radius, w)
        y_min = max(center_y - radius, 0)
        y_max = min(center_y + radius, h)

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                distance = (x - center_x)**2 + (y - center_y)**2
                accumulated_heatmap[y, x] += np.exp(-distance / (2 * (radius/2)**2))

    # Normalize
    normalized_heatmap = cv2.normalize(accumulated_heatmap, None, 0, 255, cv2.NORM_MINMAX)
    normalized_heatmap = normalized_heatmap.astype(np.uint8)

    heatmap_color = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

    return overlay