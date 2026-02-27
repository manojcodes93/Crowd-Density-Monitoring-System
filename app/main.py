from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import threading
import cv2
import time
import app.state as state
from app.engine import run_engine

app = FastAPI()

@app.on_event("startup")
def start_engine():
    thread = threading.Thread(target=run_engine)
    thread.daemon = True
    thread.start()

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>AI Crowd Detection</title>
        </head>
        <body style="background:black; color:white; text-align:center;">
            <h1>AI Crowd Detection</h1>
            <img src="/video" width="800"/>
        </body>
    </html>
    """

def generate_frames():
    while True:
        with state.lock:
            if state.output_frame is None:
                time.sleep(0.01)
                continue

            frame = state.output_frame.copy()

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buffer.tobytes() +
            b'\r\n'
        )

        time.sleep(0.03)

@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )