from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
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
            
            <h2 id="count">People: 0</h2>
            <h3 id="zones">Zones: A:0 B:0 C:0 D:0</h3>
            
            <img src="/video" width="800"/>
            
            <script>
                async function updateStats() {
                    const res = await fetch('/stats');
                    const data = await res.json();
                    
                    document.getElementById("count").innerText =
                        "People: " + data.count;

                    document.getElementById("zones").innerText =
                        "Zones: A:" + data.zones.A +
                        " B:" + data.zones.B +
                        " C:" + data.zones.C +
                        " D:" + data.zones.D;
                }

                setInterval(updateStats, 1000);
                updateStats();
            </script>
        </body>
    </html>
    """

@app.get("/stats")
def get_stats():
    with state.lock:
        return JSONResponse({
            "count": state.current_count,
            "zones": state.zones
        })

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