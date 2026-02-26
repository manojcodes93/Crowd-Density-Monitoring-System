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

# Mount frontend folder
app.mount("/static", StaticFiles(directory="frontend"), name="static")


# =========================
# START BACKGROUND ENGINE
# =========================
@app.on_event("startup")
def start_engine():
    thread = threading.Thread(target=run_engine)
    thread.daemon = True
    thread.start()


# =========================
# DASHBOARD ROUTE
# =========================
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return Path("frontend/dashboard.html").read_text(encoding="utf-8")


# =========================
# LIVE STATS
# =========================
@app.get("/stats")
def get_stats():
    return state.live_data


# =========================
# FIXED ANALYTICS ENDPOINT
# =========================
@app.get("/log-data")
def get_log_data():
    log_path = Path("crowd_log.csv")

    if not log_path.exists():
        return []

    data = []

    with open(log_path, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            try:
                data.append({
                    "timestamp": row["timestamp"],
                    "total": int(row["total"]),
                    "zoneA": int(row["zoneA"]),
                    "zoneB": int(row["zoneB"]),
                    "zoneC": int(row["zoneC"]),
                    "zoneD": int(row["zoneD"]),
                    "prediction": int(row["prediction"]),
                    "alert": row["alert"].strip().lower() == "true"
                })
            except (ValueError, KeyError):
                # Skip corrupted lines
                continue

    return JSONResponse(data)


# =========================
# VIDEO STREAM
# =========================
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

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

        time.sleep(0.03)


@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_video(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )