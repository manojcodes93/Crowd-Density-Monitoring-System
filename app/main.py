from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import threading
import cv2
import time
import csv

import app.state as state
from app.engine import run_engine

app = FastAPI()

app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ===============================
# START ENGINE
# ===============================
@app.on_event("startup")
def start_engine():
    thread = threading.Thread(target=run_engine)
    thread.daemon = True
    thread.start()


# ===============================
# DASHBOARD
# ===============================
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return Path("frontend/dashboard.html").read_text(encoding="utf-8")


# ===============================
# ANALYTICS PAGE
# ===============================
@app.get("/analytics", response_class=HTMLResponse)
def analytics_page():
    return Path("frontend/analytics.html").read_text(encoding="utf-8")


# ===============================
# STATS API
# ===============================
@app.get("/stats")
def get_stats():
    return state.live_data


# ===============================
# LOG DATA API
# ===============================
@app.get("/log-data")
def get_log_data():
    data = []
    try:
        with open("crowd_log.csv", "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except:
        pass
    return JSONResponse(data)


# ===============================
# VIDEO STREAM
# ===============================
def generate_video():
    while True:
        with state.lock:
            if state.output_frame is None:
                time.sleep(0.01)
                continue

            frame = state.output_frame.copy()

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes +
            b"\r\n"
        )

        time.sleep(0.03)


@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_video(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )