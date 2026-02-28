from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi import UploadFile, File
import threading
import cv2
import time
import csv
import shutil
import app.state as state
from app.engine import run_engine

app = FastAPI()

# ------------------ ANALYTICS API ------------------

@app.get("/analytics")
def get_analytics():
    data = []
    try:
        with open("crowd_log.csv", "r") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            data = rows[-30:]  # only last 30 entries
    except:
        pass
    return JSONResponse(data)

# ------------------ ENGINE CONTROL ------------------

CAMERA_SOURCE = 0
engine_thread = None

def start_camera(source):
    global engine_thread

    # Stop old engine safely
    if state.engine_running:
        state.engine_running = False
        time.sleep(1)

    # Start new engine
    engine_thread = threading.Thread(target=run_engine, args=(source,))
    engine_thread.daemon = True
    engine_thread.start()

@app.on_event("startup")
def startup():
    start_camera(CAMERA_SOURCE)

@app.post("/set-camera")
def set_camera(source: str):
    start_camera(source)
    return {"status": "Camera switched", "source": source}

# ------------------ DASHBOARD ------------------

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>AI Crowd Monitoring</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { background:#0e1117; color:white; font-family:Arial; text-align:center; }
                canvas { max-width:900px; margin:20px auto; display:block; }
                input { padding:8px; width:400px; }
                button { padding:8px 15px; cursor:pointer; }
                .camera-box { margin-bottom:20px; }
                .status { margin-top:10px; font-weight:bold; }
            </style>
        </head>
        <body>

            <h1>AI Crowd Monitoring Dashboard</h1>

            <div class="camera-box">
                <input type="text" id="cameraSource" placeholder="Enter 0 or RTSP URL">
                <button onclick="switchCamera()">Switch Camera</button>

                <br><br>

                <input type="file" id="videoFile">
                <button onclick="uploadVideo()">Upload Video</button>

                <div class="status" id="cameraStatus"></div>
            </div>

            <h2 id="count">People: 0</h2>
            <h3 id="prediction">Prediction: 0</h3>

            <img src="/video" width="800"/>

            <h2>Live Crowd Trend</h2>
            <canvas id="crowdChart"></canvas>

            <h2>Latest Zone Distribution</h2>
            <canvas id="zoneChart"></canvas>

            <script>

                let crowdChart = null;
                let zoneChart = null;

                async function switchCamera() {
                    const source = document.getElementById("cameraSource").value;

                    if (!source) {
                        document.getElementById("cameraStatus").innerText = "Please enter a source";
                        return;
                    }

                    document.getElementById("cameraStatus").innerText = "Switching...";

                    try {
                        const res = await fetch("/set-camera", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify(source)
                        });

                        const data = await res.json();
                        document.getElementById("cameraStatus").innerText =
                            "Connected to: " + data.source;

                    } catch (err) {
                        document.getElementById("cameraStatus").innerText = "Failed to switch camera";
                    }
                }

                async function uploadVideo() {
                    const fileInput = document.getElementById("videoFile");
                    const file = fileInput.files[0];

                    if (!file) {
                        document.getElementById("cameraStatus").innerText = "Please select a video file";
                        return;
                    }

                    document.getElementById("cameraStatus").innerText = "Uploading...";

                    const formData = new FormData();
                    formData.append("file", file);

                    try {
                        const res = await fetch("/upload-video", {
                            method: "POST",
                            body: formData
                        });

                        const data = await res.json();
                        document.getElementById("cameraStatus").innerText =
                            "Processing: " + data.file;

                    } catch (err) {
                        document.getElementById("cameraStatus").innerText = "Upload failed";
                    }
                }

                async function updateStats() {
                    const res = await fetch('/stats');
                    const data = await res.json();

                    document.getElementById("count").innerText =
                        "People: " + data.count;

                    document.getElementById("prediction").innerText =
                        "Prediction: " + data.prediction;
                }

                async function loadAnalytics() {
                    const res = await fetch('/analytics');
                    const data = await res.json();

                    if (!data || data.length === 0) return;

                    const recent = data.slice(-20);
                    const labels = recent.map(d => d.timestamp);
                    const totals = recent.map(d => Number(d.total));
                    const last = recent[recent.length - 1];

                    const ctx1 = document.getElementById("crowdChart").getContext("2d");
                    const ctx2 = document.getElementById("zoneChart").getContext("2d");

                    if (crowdChart) crowdChart.destroy();
                    if (zoneChart) zoneChart.destroy();

                    crowdChart = new Chart(ctx1, {
                        type: 'line',
                        data: {
                            labels: labels,
                            datasets: [{
                                label: 'Total Crowd',
                                data: totals,
                                borderColor: 'cyan',
                                backgroundColor: 'rgba(0,255,255,0.2)',
                                fill: true,
                                tension: 0.3
                            }]
                        },
                        options: { scales: { y: { beginAtZero: true } } }
                    });

                    zoneChart = new Chart(ctx2, {
                        type: 'bar',
                        data: {
                            labels: ['Zone A','Zone B','Zone C','Zone D'],
                            datasets: [{
                                label: 'Latest Zones',
                                data: [
                                    Number(last.zoneA),
                                    Number(last.zoneB),
                                    Number(last.zoneC),
                                    Number(last.zoneD)
                                ],
                                backgroundColor: ['cyan','yellow','red','lime']
                            }]
                        },
                        options: { scales: { y: { beginAtZero: true } } }
                    });
                }

                setInterval(updateStats, 1000);
                setInterval(loadAnalytics, 5000);

                updateStats();
                loadAnalytics();

            </script>

        </body>
    </html>
    """

# ------------------ STATS API ------------------

@app.get("/stats")
def get_stats():
    with state.lock:
        return {
            "count": state.current_count,
            "zones": state.zones,
            "prediction": state.prediction,
            "alert": state.alert
        }

# ------------------ VIDEO STREAM ------------------

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

# ------------------ VIDEO UPLOAD ------------------

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    upload_path = f"uploaded_{file.filename}"

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    start_camera(upload_path)

    return {"status": "Video uploaded and processing started", "file": upload_path}