import numpy as np
import cv2

def generate_heatmap(frame, boxes):
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    for (x1, y1, x2, y2) in boxes:
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Create Gaussian blob around center
        radius = 40  # controls blob size

        x_min = max(center_x - radius, 0)
        x_max = min(center_x + radius, frame.shape[1])
        y_min = max(center_y - radius, 0)
        y_max = min(center_y + radius, frame.shape[0])

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                distance = (x - center_x)**2 + (y - center_y)**2
                heatmap[y, x] += np.exp(-distance / (2 * (radius/2)**2))

    # Normalize
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

    return overlay
