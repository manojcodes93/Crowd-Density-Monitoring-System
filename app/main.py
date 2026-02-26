from fastapi import FastAPI
from app.state import live_data
from app.engine import run_engine
import threading

app = FastAPI()

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