def calculate_zones(boxes, frame_shape):
    height, width, _ = frame_shape

    mid_x = width // 2
    mid_y = height // 2

    zones = {
        "A": 0,
        "B": 0,
        "C": 0,
        "D": 0
    }

    for (x1, y1, x2, y2) in boxes:
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if center_x < mid_x and center_y < mid_y:
            zones["A"] += 1
        elif center_x >= mid_x and center_y < mid_y:
            zones["B"] += 1
        elif center_x < mid_x and center_y >= mid_y:
            zones["C"] += 1
        else:
            zones["D"] += 1

    return zones
