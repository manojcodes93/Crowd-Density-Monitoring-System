import threading

live_data = {
    "total": 0,
    "zoneA": 0,
    "zoneB": 0,
    "zoneC": 0,
    "zoneD": 0,
    "prediction": 0,
    "alert": False
}

# Shared video frame
output_frame = None

# Shared lock
lock = threading.Lock()