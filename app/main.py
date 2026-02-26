from fastapi import FastAPI
from app.state import live_data
from app.engine import run_engine
import threading
from fastapi.responses import StreamingResponse
import cv2
from app.engine import output_frame, lock
import time

app = FastAPI()

def generate_video():
    while True:
        with lock:
            if output_frame is None:
                time.sleep(0.01)
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

@app.get("/video")
def video_feed():
    return StreamingResponse(generate_video(),
        media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
def root():
    return {"message": "Crowd Monitoring API is running"}

# Start detection in background
@app.on_event("startup")
def start_engine():
    print("Starting detection engine...")
    thread = threading.Thread(target=run_engine)
    thread.daemon = True
    thread.start()

@app.get("/stats")
def get_stats():
    return live_data