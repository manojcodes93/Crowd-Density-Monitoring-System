from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
import threading
import cv2
import time
import shutil
import app.state as state
from app.engine import run_engine
from app.database import init_db, get_last_logs

app = FastAPI()

CAMERA_SOURCE = 0
engine_thread = None


# ------------------ ENGINE CONTROL ------------------

def start_camera(source):
    global engine_thread

    if isinstance(source, str) and source.isdigit():
        source = int(source)

    if state.engine_running:
        state.engine_running = False
        time.sleep(1)

    engine_thread = threading.Thread(target=run_engine, args=(source,))
    engine_thread.daemon = True
    engine_thread.start()


@app.on_event("startup")
def startup():
    init_db()
    start_camera(CAMERA_SOURCE)


@app.post("/set-camera")
def set_camera(source: str):
    start_camera(source)
    return {"status": "Camera switched", "source": source}


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    upload_path = "uploaded_video.mp4"

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    start_camera(upload_path)

    return {"status": "Video uploaded", "file": upload_path}


# ------------------ STATS API ------------------

@app.get("/stats")
def get_stats():
    with state.lock:
        return {
            "count": state.current_count,
            "zones": state.zones,
            "prediction": state.prediction,
            "alert": state.alert,
            "status": state.status
        }


@app.get("/analytics")
def get_analytics():
    return get_last_logs(30)


# ------------------ VIDEO STREAM ------------------

def generate_frames():
    while True:
        with state.lock:
            if state.output_frame is None:
                continue
            frame = state.output_frame.copy()

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buffer.tobytes() +
            b'\r\n'
        )


@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


# ------------------ DASHBOARD ------------------

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
<title>AI Crowd Monitoring</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body { margin:0; font-family:'Segoe UI', sans-serif; background:#0f172a; color:white; }
.navbar { background:#111827; padding:15px 30px; font-size:22px; font-weight:bold; }
.container { display:flex; padding:20px; gap:20px; }
.left-panel { width:300px; background:#1e293b; padding:20px; border-radius:10px; }
.right-panel { flex:1; background:#1e293b; padding:20px; border-radius:10px; }
input,button { width:100%; padding:10px; margin-top:10px; border-radius:6px; border:none; }
input { background:#334155; color:white; }
button { background:#2563eb; color:white; font-weight:bold; cursor:pointer; }
button:hover { background:#1d4ed8; }
.status-box { margin-top:20px; padding:15px; background:#0f172a; border-radius:8px; }
.stats { display:flex; gap:20px; margin-bottom:20px; }
.card { flex:1; background:#0f172a; padding:20px; border-radius:10px; text-align:center; }
img { width:100%; border-radius:10px; margin-top:15px; }
canvas { margin-top:25px; }
</style>
</head>

<body>

<div class="navbar">AI Crowd Monitoring Dashboard</div>

<div class="container">

<div class="left-panel">
<h3>Camera Controls</h3>

<input type="text" id="cameraSource" placeholder="Enter 0 or RTSP URL">
<button onclick="switchCamera()">Switch Camera</button>

<input type="file" id="videoFile">
<button onclick="uploadVideo()">Upload Video</button>

<div class="status-box">
<strong>Status:</strong> <span id="statusText">Connected</span>
</div>
</div>

<div class="right-panel">

<div class="stats">
<div class="card">
<h2 id="count">0</h2>
<p>People Detected</p>
</div>

<div class="card">
<h2 id="prediction">0</h2>
<p>Prediction</p>
</div>

<div class="card">
<h2 id="alert">NORMAL</h2>
<p>System Status</p>
</div>
</div>

<img id="videoStream" src="/video">

<h3>Live Crowd Trend</h3>
<canvas id="crowdChart"></canvas>

<h3>Zone Distribution</h3>
<canvas id="zoneChart"></canvas>

</div>
</div>

<script>

let crowdChart = null;
let zoneChart = null;

async function switchCamera() {
    const source = document.getElementById("cameraSource").value;
    if (!source) return;

    document.getElementById("statusText").innerText = "Switching...";
    await fetch("/set-camera?source=" + source, { method: "POST" });

    document.getElementById("videoStream").src = "/video?ts=" + new Date().getTime();
    document.getElementById("statusText").innerText = "Connected";
}

async function uploadVideo() {
    const fileInput = document.getElementById("videoFile");
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    await fetch("/upload-video", {
        method: "POST",
        body: formData
    });

    document.getElementById("videoStream").src = "/video?ts=" + new Date().getTime();
    document.getElementById("statusText").innerText = "Video Loaded";
}

async function updateStats() {
    const res = await fetch('/stats');
    const data = await res.json();

    document.getElementById("count").innerText = data.count;
    document.getElementById("prediction").innerText = data.prediction;
    document.getElementById("alert").innerText = data.status;

    if (data.status === "CRITICAL")
        alert.style.color = "#ef4444";
    else if (data.status === "MODERATE")
        alert.style.color = "#facc15";
    else
        alert.style.color = "#22c55e";
}

async function loadAnalytics() {
    const res = await fetch('/analytics');
    const data = await res.json();
    if (!data.length) return;

    const labels = data.map(d => d.timestamp);
    const totals = data.map(d => Number(d.total));
    const last = data[data.length - 1];

    const ctx1 = document.getElementById("crowdChart").getContext("2d");
    const ctx2 = document.getElementById("zoneChart").getContext("2d");

    if (crowdChart) crowdChart.destroy();
    if (zoneChart) zoneChart.destroy();

    crowdChart = new Chart(ctx1, {
        type: 'line',
        data: { labels: labels, datasets: [{ label:'Total Crowd', data: totals, borderColor:'#38bdf8', fill:true }] },
        options: { scales:{ y:{ beginAtZero:true } } }
    });

    zoneChart = new Chart(ctx2, {
        type: 'bar',
        data: {
            labels:['Zone A','Zone B','Zone C','Zone D'],
            datasets:[{
                label:'Latest Zones',
                data:[last.zoneA,last.zoneB,last.zoneC,last.zoneD]
            }]
        },
        options:{ scales:{ y:{ beginAtZero:true } } }
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