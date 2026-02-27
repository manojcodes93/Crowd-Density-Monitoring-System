import threading

output_frame = None
current_count = 0

zones = {
    "A": 0,
    "B": 0,
    "C": 0,
    "D": 0
}

lock = threading.Lock()