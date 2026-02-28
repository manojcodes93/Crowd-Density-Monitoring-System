import threading

lock = threading.Lock()

output_frame = None
current_count = 0

zones = {
    "A": 0,
    "B": 0,
    "C": 0,
    "D": 0
}

prediction = 0
alert = False

camera_source = 0
engine_running = False